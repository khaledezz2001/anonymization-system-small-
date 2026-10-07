import runpod
import re
import json
import os

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# ===============================
# MODEL CONFIG (loaded inside __main__ guard)
# ===============================
MODEL_PATH = "/app/models/Qwen3-8B"

# These globals are set inside if __name__ == '__main__' before any
# function is called.  Declared here so linters don't complain.
tokenizer = None
llm = None


# ===============================
# CORE FUNCTIONS
# ===============================
def combine_pages(pages):
    sorted_pages = sorted(pages, key=lambda p: p["page"])
    return "\n\n".join(p["text"] for p in sorted_pages)


def chunk_text_with_overlap(text, max_tokens=1500, overlap_tokens=150):
    """Split text into token-limited chunks with sentence-boundary awareness.

    Used ONLY for individual pages that exceed the token limit.
    Most pages fit in a single chunk and are processed as-is.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        # Page fits in one chunk — return as-is
        return [text.strip()] if text.strip() else []

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_text = tokenizer.decode(tokens[start:end], skip_special_tokens=True)
        actual_end = end

        # Try to break at sentence boundary if not at the end of the text
        if end < len(tokens) and len(chunk_text) > 200:
            sentence_enders = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
            best_break = -1
            for ender in sentence_enders:
                idx = chunk_text.rfind(ender)
                if idx > best_break:
                    best_break = idx

            # Only break at sentence if boundary is in the last 40% of chunk
            min_break_pos = len(chunk_text) * 6 // 10
            if best_break >= min_break_pos:
                trimmed_text = chunk_text[:best_break + 1].strip()
                if trimmed_text:
                    trimmed_tokens = tokenizer.encode(trimmed_text, add_special_tokens=False)
                    chunk_text = trimmed_text
                    actual_end = start + len(trimmed_tokens)

        chunk_text = chunk_text.strip()
        if chunk_text:
            chunks.append(chunk_text)
        if actual_end >= len(tokens):
            break
        start = max(actual_end - overlap_tokens, start + 1)

    return chunks


def pages_to_chunks(pages, max_tokens=1500):
    """Convert pages to chunks for LLM processing.

    Each page is kept as its own chunk to preserve natural document structure
    (headers, signature blocks, tables stay intact). Only pages that exceed
    the token limit are split into sub-chunks.
    """
    sorted_pages = sorted(pages, key=lambda p: p["page"])
    chunks = []
    for page in sorted_pages:
        text = page["text"].strip()
        if not text:
            continue
        page_chunks = chunk_text_with_overlap(text, max_tokens=max_tokens)
        chunks.extend(page_chunks)
    return chunks


MAX_CHUNKS = 120   # safety limit for very large documents


# ===============================
# STAGE-SPECIFIC SYSTEM PROMPTS
# ===============================
STAGE1_SYSTEM_PROMPT = """You are a multilingual named entity recognition (NER) assistant for legal and business documents.
You MUST extract entities in ALL languages and scripts, including but not limited to: English, Russian (Cyrillic), Greek, Arabic, French, German, Turkish, and any other language present.

Extract the following from the text:

1. PERSONS — actual human names ONLY
   ✓ EXTRACT: "John Smith", "Andreas Menelaou", "Борис Грановский", "Γεώργιος Τσιφραρίδης", "В.А. Король"
   ✗ NEVER extract role titles or descriptions as persons. These are NOT persons:
     - "Chairman", "Director", "Secretary", "Landlord", "Tenant", "Auditor"
     - "the auditor for the time being of the Company"
     - "the Chairman of the Board"
     - Any phrase starting with "the " followed by a role — this is a description, NOT a name
   - Extract person names in ALL scripts and languages
   - Extract names EXACTLY as they appear in the text, preserving the EXACT grammatical form/case
     In Russian: if text says "Иванова Ивана Ивановича" (genitive), extract that exact form
   - Extract person names from witnesses, signatories, advocates, directors, shareholders
   - If a person's name is used as a business/firm name, extract it as BOTH a person AND an organisation
   - Extract ALL variants/transliterations of the same person

2. ORGANISATIONS — actual registered business entity names ONLY
   An organisation MUST be a specific, named, registered business entity (a company, LLC, partnership, etc.)
   It MUST have a proper name — typically ending in Ltd, Limited, LLC, Inc, Corp, GmbH, S.A., etc.
   ✓ EXTRACT: "Altus Citadel Corporate Services Limited", "ООО Ромашка", "PROSPERITY CHAIN TECHNOLOGY CO., LIMITED"
   ✗ NEVER extract these as organisations:
     - Generic terms: "the Company", "Company", "Board of Directors", "the Board"
     - Role references: "the Option Holder", "the Option Issuer", "the Seller", "the Buyer", "the Lender"
     - Legal/financial terms: "Call Option", "Put Option", "Exercise Price", "Option Shares", "Option Deed", "Completion", "Affiliates", "Bank account"
     - Document clause definitions: any capitalized term that is DEFINED in the document as a concept (not a company name)
     - Government bodies: "European Commission", "FATF", "MOKAS", "United Nations"
     - Regulations/rules: "Directive (EU) 2018/843", "GDPR", "EBA Guidelines", "LCIA Rules", "ICC Rules"
     - Countries/areas: "BVI", "European Economic Area"
     - Indices/reports: "Basel AML Index"
   CRITICAL: If the text defines a term with a capital letter (e.g., "Call Option" means..., "the Company" means...), that is a DEFINED TERM, NOT an organisation. Do NOT extract it.

Output ONLY valid JSON with no explanation. Do not wrap in markdown code blocks.

{
  "persons": ["name1", "name2"],
  "organizations": ["org1", "org2"]
}"""


STAGE2_SYSTEM_PROMPT = """You are a multilingual named entity recognition (NER) assistant for legal and business documents.
You MUST extract entities in ALL languages and scripts, including but not limited to: English, Russian (Cyrillic), Greek, Arabic, French, German, Turkish, and any other language present.

Extract the following from the text:

1. DATES — specific, concrete calendar dates ONLY (must identify a SPECIFIC day on a calendar)
   ✓ EXTRACT: "01/09/2015", "24th of July, 2015", "1 January 2020", "20 December 2018"
   ✗ NEVER extract these as dates:
     - Time durations: "fourteen days", "six months", "ten days", "twenty-one days", "3 months", "1 year", "two weeks", "no later than 10 days as from the request"
     - Bare years: "2014" alone is NOT a date
     - Quarter references: "Q2 2024"
     - Section/article numbers: "2.2.11", "3.1.5"
     - Times of day (without a date): "11.00 a.m.", "11:00 a.m.", "2:30 p.m." — these are TIMES, not dates
     - Template placeholders with blanks: "DATED this _____", "this ____ day of", "the day and year first above written" — these have NO specific date
     - Placeholder references in brackets: "[Completion Date]", "[Date]", "[insert date]" — these are placeholders, NOT dates
     - Relative date descriptions: "the date of expiry of the Exercise Period", "1st anniversary of this Deed", "a date falling 5 Business Days after" — these DESCRIBE a concept, they are NOT specific dates
     - Descriptive clauses about dates: "date that the respective obligations...has terminated" — this is a legal definition, NOT a date
     - Complex temporal phrases: "11:00 a.m. (UTC time) upon a date falling 5 (five) Business Days after" — this is a calculation rule, NOT a date
   CRITICAL RULE: A date MUST be a SPECIFIC point in time that you could mark on a calendar (e.g., "20 December 2018"). If you cannot identify the exact day, month, or year, do NOT extract it.

2. ADDRESSES — physical street/postal addresses in any language
   ✓ EXTRACT: "Mome Kapora 12, apartment 11, 1100 Belgrade", "191 ATHALASSIS AVE."
   ✗ NEVER extract these as addresses:
     - Page numbers or section headers: "4 INTRODUCTION"
     - Duration phrases: "1 year for high risk customers"
     - Counts: "2 clients onboarded"
     - Legal references: "8 and Chapter VI of Directive"
   - Extract the FULL address as a SINGLE string (street + number + apartment + postal code + city + country)
   - If an address spans multiple lines, combine ALL lines into one address string
   - Addresses can be in ANY format and ANY language
   - Even PARTIAL addresses are PII: "Eleftherias 5" alone is an address
   - When in doubt about whether something is an address, extract it

Output ONLY valid JSON with no explanation. Do not wrap in markdown code blocks.

{
  "dates": ["date1", "date2"],
  "addresses": ["addr1", "addr2"]
}"""


STAGE3_SYSTEM_PROMPT = """You are a multilingual named entity recognition (NER) assistant for legal and business documents.
You MUST extract entities in ALL languages and scripts, including but not limited to: English, Russian (Cyrillic), Greek, Arabic, French, German, Turkish, and any other language present.

Extract the following from the text:

1. PHONES — phone and fax numbers
   ✓ EXTRACT: "+357 22 315161", "22314641"
   ✗ NEVER extract bank account numbers, IBAN codes, or registration/tax numbers (ИНН, ОГРН, КПП) as phones

2. REGISTRATION IDS — company registration numbers, tax IDs
   ✓ EXTRACT: "H.E.107777", "HE317807", "Company No. 12345678"
   ✗ NEVER extract template placeholders like "Company Number: [ - ]" or "Company No. [____]" — these contain NO actual number

3. BANK ACCOUNTS — actual IBAN numbers, bank account numbers, SWIFT/BIC codes
   ✓ EXTRACT: "CY17 0020 0128 0000 0012 0052 7600", "BCYPCY2N"
   ✗ NEVER extract template placeholders or instructions: "[insert details of the bank account]", "[bank account number]" — these are NOT actual bank accounts

4. EMAILS — email addresses ONLY (MUST contain an @ symbol)
   ✓ EXTRACT: "john@example.com", "info@company.com"
   ✗ NEVER extract the bare word "email" or "Email" — only extract actual email addresses with @ symbol
   ✗ NEVER extract descriptions about email: "e-mail designated by the Option Holder" is NOT an email address
   ✗ NEVER extract URLs or domain names: "www.example.com" is NOT an email
   CRITICAL: An email MUST contain the @ character. If there is no @, it is NOT an email.

5. PASSPORTS — passport numbers, national ID numbers, travel document numbers
   ✓ EXTRACT: "N1234567", "C12345678"
   ✗ NEVER extract section/article numbers as passport numbers

Output ONLY valid JSON with no explanation. Do not wrap in markdown code blocks.

{
  "phones": ["phone1", "phone2"],
  "registration_ids": ["H.E.107777"],
  "bank_accounts": ["CY17 0020 0128 0000 0012 0052 7600"],
  "emails": ["email1@example.com"],
  "passports": ["N1234567"]
}"""


# Legacy single-pass prompt (used when custom system_prompt is provided)
LEGACY_SYSTEM_PROMPT = """You are a multilingual named entity recognition (NER) assistant for legal and business documents.
You MUST extract entities in ALL languages and scripts, including but not limited to: English, Russian (Cyrillic), Greek, Arabic, French, German, Turkish, and any other language present.

Extract ALL of the following from the text:

1. PERSONS — actual human names ONLY
   ✓ EXTRACT: "John Smith", "Andreas Menelaou", "Борис Грановский", "Γεώργιος Τσιφραρίδης", "В.А. Король"
   ✗ NEVER extract role titles or descriptions as persons. These are NOT persons:
     - "Chairman", "Director", "Secretary", "Landlord", "Tenant", "Auditor"
     - "the auditor for the time being of the Company"
     - "the Chairman of the Board"
     - Any phrase starting with "the " followed by a role — this is a description, NOT a name
   - Extract person names in ALL scripts and languages
   - Extract names EXACTLY as they appear in the text, preserving the EXACT grammatical form/case
     In Russian: if text says "Иванова Ивана Ивановича" (genitive), extract that exact form
   - Extract person names from witnesses, signatories, advocates, directors, shareholders
   - If a person's name is used as a business/firm name, extract it as BOTH a person AND an organisation
   - Extract ALL variants/transliterations of the same person

2. ORGANISATIONS — actual registered business entity names ONLY
   An organisation MUST be a specific, named, registered business entity (a company, LLC, partnership, etc.)
   It MUST have a proper name — typically ending in Ltd, Limited, LLC, Inc, Corp, GmbH, S.A., etc.
   ✓ EXTRACT: "Altus Citadel Corporate Services Limited", "ООО Ромашка", "PROSPERITY CHAIN TECHNOLOGY CO., LIMITED"
   ✗ NEVER extract these as organisations:
     - Generic terms: "the Company", "Company", "Board of Directors", "the Board"
     - Role references: "the Option Holder", "the Option Issuer", "the Seller", "the Buyer", "the Lender"
     - Legal/financial terms: "Call Option", "Put Option", "Exercise Price", "Option Shares", "Option Deed", "Completion", "Affiliates", "Bank account"
     - Document clause definitions: any capitalized term that is DEFINED in the document as a concept (not a company name)
     - Government bodies: "European Commission", "FATF", "MOKAS", "United Nations"
     - Regulations/rules: "Directive (EU) 2018/843", "GDPR", "EBA Guidelines", "LCIA Rules", "ICC Rules"
     - Countries/areas: "BVI", "European Economic Area"
     - Indices/reports: "Basel AML Index"
   CRITICAL: If the text defines a term with a capital letter (e.g., "Call Option" means..., "the Company" means...), that is a DEFINED TERM, NOT an organisation. Do NOT extract it.

3. DATES — specific, concrete calendar dates ONLY (must identify a SPECIFIC day on a calendar)
   ✓ EXTRACT: "01/09/2015", "24th of July, 2015", "1 January 2020", "20 December 2018"
   ✗ NEVER extract these as dates:
     - Time durations: "fourteen days", "six months", "ten days", "twenty-one days", "3 months", "1 year", "two weeks", "no later than 10 days as from the request"
     - Bare years: "2014" alone is NOT a date
     - Quarter references: "Q2 2024"
     - Section/article numbers: "2.2.11", "3.1.5"
     - Times of day (without a date): "11.00 a.m.", "11:00 a.m.", "2:30 p.m." — these are TIMES, not dates
     - Template placeholders with blanks: "DATED this _____", "this ____ day of", "the day and year first above written"
     - Placeholder references in brackets: "[Completion Date]", "[Date]", "[insert date]"
     - Relative date descriptions: "the date of expiry of the Exercise Period", "1st anniversary of this Deed", "a date falling 5 Business Days after"
     - Descriptive clauses about dates: "date that the respective obligations...has terminated"
     - Complex temporal phrases: "11:00 a.m. (UTC time) upon a date falling 5 (five) Business Days after"
   CRITICAL RULE: A date MUST be a SPECIFIC point in time that you could mark on a calendar. If you cannot identify the exact day, month, or year, do NOT extract it.

4. ADDRESSES — physical street/postal addresses in any language
   ✓ EXTRACT: "Mome Kapora 12, apartment 11, 1100 Belgrade", "191 ATHALASSIS AVE."
   ✗ NEVER extract these as addresses:
     - Page numbers or section headers: "4 INTRODUCTION"
     - Duration phrases: "1 year for high risk customers"
     - Counts: "2 clients onboarded"
     - Legal references: "8 and Chapter VI of Directive"
   - Extract the FULL address as a SINGLE string (street + number + apartment + postal code + city + country)
   - If an address spans multiple lines, combine ALL lines into one address string
   - Addresses can be in ANY format and ANY language
   - Even PARTIAL addresses are PII: "Eleftherias 5" alone is an address
   - When in doubt about whether something is an address, extract it

5. PHONES — phone and fax numbers
   ✓ EXTRACT: "+357 22 315161", "22314641"
   ✗ NEVER extract bank account numbers, IBAN codes, or registration/tax numbers (ИНН, ОГРН, КПП) as phones

6. REGISTRATION IDS — company registration numbers, tax IDs
   ✓ EXTRACT: "H.E.107777", "HE317807", "Company No. 12345678"
   ✗ NEVER extract template placeholders like "Company Number: [ - ]" or "Company No. [____]"

7. BANK ACCOUNTS — actual IBAN numbers, bank account numbers, SWIFT/BIC codes
   ✓ EXTRACT: "CY17 0020 0128 0000 0012 0052 7600", "BCYPCY2N"
   ✗ NEVER extract template placeholders or instructions: "[insert details of the bank account]", "[bank account number]"

8. EMAILS — email addresses ONLY (MUST contain an @ symbol)
   ✓ EXTRACT: "john@example.com", "info@company.com"
   ✗ NEVER extract the bare word "email" or "Email" — only extract actual email addresses with @ symbol
   ✗ NEVER extract descriptions about email: "e-mail designated by the Option Holder" is NOT an email address
   ✗ NEVER extract URLs or domain names: "www.example.com" is NOT an email
   CRITICAL: An email MUST contain the @ character. If there is no @, it is NOT an email.

9. PASSPORTS — passport numbers, national ID numbers, travel document numbers
   ✓ EXTRACT: "N1234567", "C12345678"
   ✗ NEVER extract section/article numbers as passport numbers

Output ONLY valid JSON with no explanation. Do not wrap in markdown code blocks.

{
  "persons": ["name1", "name2"],
  "organizations": ["org1", "org2"],
  "dates": ["date1", "date2"],
  "addresses": ["addr1", "addr2"],
  "phones": ["phone1", "phone2"],
  "registration_ids": ["H.E.107777"],
  "bank_accounts": ["CY17 0020 0128 0000 0012 0052 7600"],
  "emails": ["email1@example.com"],
  "passports": ["N1234567"]
}"""


# Pipeline stage definitions: (system_prompt, default_user_prompt, expected_keys, max_tokens)
PIPELINE_STAGES = [
    (
        STAGE1_SYSTEM_PROMPT,
        "Extract all person names and organization names from the following text:",
        ["persons", "organizations"],
        2048,
    ),
    (
        STAGE2_SYSTEM_PROMPT,
        "Extract all dates and physical addresses from the following text:",
        ["dates", "addresses"],
        2048,
    ),
    (
        STAGE3_SYSTEM_PROMPT,
        "Extract all phone numbers, company registration numbers, bank account numbers, email addresses, and passport numbers from the following text:",
        ["phones", "registration_ids", "bank_accounts", "emails", "passports"],
        2048,
    ),
]


def strip_thinking(text):
    """Remove <think>...</think> blocks from model output (safety net)."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    return text.strip()


def _run_extraction_stage(chunks, system_prompt, default_user_prompt, entity_keys,
                          max_tokens, user_prompt_override=None):
    """Run a single extraction stage on all chunks.

    Args:
        chunks: List of text chunks to process.
        system_prompt: The stage-specific system prompt.
        default_user_prompt: Default user prompt for this stage.
        entity_keys: List of JSON keys expected in the output (e.g. ["persons", "organizations"]).
        max_tokens: Max tokens for generation in this stage.
        user_prompt_override: Optional user-provided prompt override.

    Returns:
        dict mapping each entity_key to a list of extracted values,
        plus a "custom" key for any unexpected keys.
    """
    BATCH_SIZE = 24

    stage_sampling = SamplingParams(
        temperature=0,
        max_tokens=max_tokens,
        repetition_penalty=1.1,
    )

    results = {key: [] for key in entity_keys}
    results["custom"] = []

    total_chunks = len(chunks)
    for batch_start in range(0, total_chunks, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_chunks)
        batch_chunks = chunks[batch_start:batch_end]

        print(f"[LOG]   Chunks {batch_start + 1}-{batch_end} of {total_chunks}", flush=True)

        prompts = []
        for chunk in batch_chunks:
            if user_prompt_override and isinstance(user_prompt_override, str) and user_prompt_override.strip():
                if "{chunk}" in user_prompt_override:
                    user_content = user_prompt_override.replace("{chunk}", chunk)
                else:
                    user_content = f"{user_prompt_override.strip()}\n\n{chunk}"
            else:
                user_content = f"{default_user_prompt}\n\n{chunk}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            prompts.append(prompt)

        outputs = llm.generate(prompts, stage_sampling)

        for output in outputs:
            raw = output.outputs[0].text.strip()
            cleaned = strip_thinking(raw)

            try:
                # Try to find a JSON object with any of the expected keys
                first_key = entity_keys[0]
                json_match = re.search(
                    r'\{[^{}]*"' + re.escape(first_key) + r'"\s*:.*\}', cleaned, re.DOTALL
                )
                if not json_match:
                    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                result = json.loads(json_match.group()) if json_match else json.loads(cleaned)

                for key in entity_keys:
                    values = result.get(key, [])
                    # Handle passports alternate key
                    if key == "passports" and not values:
                        values = result.get("passport_numbers", [])
                    if isinstance(values, list):
                        results[key].extend(
                            [v.strip() for v in values if v and isinstance(v, str) and v.strip()]
                        )

                # Collect any custom/unexpected keys
                expected_keys_set = set(entity_keys) | {"passport_numbers"}
                for k, v in result.items():
                    if k not in expected_keys_set and isinstance(v, list):
                        results["custom"].extend(
                            [item.strip() for item in v if item and isinstance(item, str) and item.strip()]
                        )
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"[WARN] Failed to parse chunk output: {e}", flush=True)
                print(f"[WARN] Raw output was: {raw[:500]}", flush=True)

    return results


def extract_entities_batch(chunks, system_prompt=None, user_prompt=None):
    """Extract entities from all chunks using a multi-stage pipeline.

    When no custom system_prompt is provided, runs 3 focused stages:
      Stage 1: Persons & Organizations
      Stage 2: Dates & Addresses
      Stage 3: Phones, Registration IDs, Bank Accounts, Emails, Passports

    When a custom system_prompt is provided, falls back to single-pass
    extraction for backward compatibility.
    """
    if system_prompt:
        # ---- LEGACY SINGLE-PASS MODE (custom prompt) ----
        print("[LOG] Custom system prompt detected — using single-pass extraction", flush=True)
        all_keys = ["persons", "organizations", "dates", "addresses",
                     "phones", "registration_ids", "bank_accounts", "emails", "passports"]
        stage_results = _run_extraction_stage(
            chunks, system_prompt,
            default_user_prompt="Extract all entities specified in the system prompt:",
            entity_keys=all_keys,
            max_tokens=4096,
            user_prompt_override=user_prompt,
        )
    else:
        # ---- MULTI-STAGE PIPELINE ----
        stage_results = {}
        all_custom = []
        for stage_idx, (stage_prompt, stage_user_prompt, stage_keys, stage_max_tokens) in enumerate(PIPELINE_STAGES, 1):
            print(f"[LOG] === Stage {stage_idx}/3: extracting {', '.join(stage_keys)} ===", flush=True)
            result = _run_extraction_stage(
                chunks, stage_prompt,
                default_user_prompt=stage_user_prompt,
                entity_keys=stage_keys,
                max_tokens=stage_max_tokens,
                user_prompt_override=user_prompt,
            )
            # Accumulate custom entities before update() overwrites the key
            all_custom.extend(result.pop("custom", []))
            stage_results.update(result)
        stage_results["custom"] = all_custom

    all_persons = stage_results.get("persons", [])
    all_orgs = stage_results.get("organizations", [])
    all_dates = stage_results.get("dates", [])
    all_addresses = stage_results.get("addresses", [])
    all_phones = stage_results.get("phones", [])
    all_reg_ids = stage_results.get("registration_ids", [])
    all_bank_accounts = stage_results.get("bank_accounts", [])
    all_emails = stage_results.get("emails", [])
    all_passports = stage_results.get("passports", [])
    custom_entities = stage_results.get("custom", [])

    print(f"[LOG] Entity extraction complete. Found: {len(all_persons)} persons, "
          f"{len(all_orgs)} orgs, {len(all_dates)} dates, {len(all_addresses)} addresses, "
          f"{len(all_phones)} phones, {len(all_reg_ids)} reg_ids, {len(all_bank_accounts)} bank_accounts, "
          f"{len(all_emails)} emails, {len(all_passports)} passports, {len(custom_entities)} custom_entities",
          flush=True)

    return all_persons, all_orgs, all_dates, all_addresses, all_phones, all_reg_ids, all_bank_accounts, all_emails, all_passports, custom_entities


# ===============================
# ANTI-HALLUCINATION VALIDATION
# ===============================
def verify_entity_in_text(entity, full_text_lower):
    """Check if an entity actually exists in the source document.

    Uses case-insensitive matching with flexible whitespace.
    For multi-word entities (like person names), also checks if all individual
    words appear nearby in the text — this handles Russian name inflections
    where the model might extract a slightly different form.
    Returns True if the entity (or a whitespace-flexible variant) is found.
    """
    entity_clean = entity.strip()
    if not entity_clean:
        return False

    # Fast path: direct case-insensitive substring match
    if entity_clean.lower() in full_text_lower:
        return True

    # Flexible whitespace: the entity might span a line break in the source
    # e.g., model says "John Smith" but text has "John\nSmith"
    words = entity_clean.split()
    if len(words) > 1:
        flexible = r'\s+'.join(re.escape(w) for w in words)
        if re.search(flexible, full_text_lower, re.IGNORECASE):
            return True

    # Stem-aware check for inflected languages (Russian, Greek, etc.):
    # If the entity has multiple words and each word's stem (first 3+ chars)
    # appears within a reasonable window in the text, accept it.
    # This catches cases like model returning "Иванов Иван Иванович"
    # when text has "Иванова Ивана Ивановича" (different grammatical case).
    if len(words) >= 2:
        # Check if all words (or their stems) appear in the text
        all_words_found = True
        for word in words:
            word_lower = word.lower().rstrip('.,;:')
            if len(word_lower) < 2:
                continue  # skip initials like "В." — too short to verify
            if word_lower in full_text_lower:
                continue
            # Try stem match: first N chars (min 3) to handle inflection
            stem_len = max(3, len(word_lower) - 2)
            stem = word_lower[:stem_len]
            if stem in full_text_lower:
                continue
            all_words_found = False
            break
        if all_words_found:
            return True

    return False


def validate_entities(entities, full_text_lower, entity_type):
    """Filter out hallucinated entities that don't appear in the source text."""
    valid = []
    removed = []
    for entity in entities:
        if verify_entity_in_text(entity, full_text_lower):
            valid.append(entity)
        else:
            removed.append(entity)

    if removed:
        print(f"[HALLUCINATION] Removed {len(removed)} fake {entity_type}: {removed[:10]}", flush=True)

    return valid


# ===============================
# POST-EXTRACTION FALSE-POSITIVE FILTERS
# ===============================

# --- Organisation false-positive filter ---
# Common legal/financial defined terms that are NOT company names.
# These are matched case-insensitively.
_ORG_BLACKLIST_EXACT = {
    "the company", "company", "affiliates", "affiliate",
    "bank account", "call option", "put option",
    "exercise price", "option shares", "option deed",
    "completion", "the board", "board of directors",
    "the board of directors", "the option holder",
    "the option issuer", "the seller", "the buyer",
    "the lender", "the borrower", "the landlord", "the tenant",
    "the grantor", "the grantee", "the licensor", "the licensee",
    "the agent", "the trustee", "the beneficiary",
    "the shareholder", "the subscriber", "the investor",
    "the creditor", "the debtor", "the guarantor",
    "lcia rules", "icc rules", "uncitral rules",
}

# Patterns that indicate an org entry is actually a defined term / role reference
_ORG_BLACKLIST_PATTERNS = [
    re.compile(r'^the\s+(?:option|call|put|exercise|completion|share)', re.IGNORECASE),
    re.compile(r'^the\s+board\b', re.IGNORECASE),
    re.compile(r'^board\s+of\s+directors', re.IGNORECASE),
    re.compile(r'^the\s+\w+\s+of\s+\[-?\]', re.IGNORECASE),  # "The Board of Directors of [-]"
]


def filter_false_positive_orgs(orgs):
    """Remove generic legal/financial terms from the organisations list."""
    filtered = []
    removed = []
    for org in orgs:
        org_lower = org.strip().lower()
        # Exact blacklist match
        if org_lower in _ORG_BLACKLIST_EXACT:
            removed.append(org)
            continue
        # Pattern-based match
        if any(p.search(org) for p in _ORG_BLACKLIST_PATTERNS):
            removed.append(org)
            continue
        filtered.append(org)
    if removed:
        print(f"[FILTER] Removed {len(removed)} false-positive orgs: {removed}", flush=True)
    return filtered


# --- Date false-positive filter ---
_DATE_BLACKLIST_PATTERNS = [
    # Template placeholders with blanks / underscores
    re.compile(r'___|\[.*?\]', re.IGNORECASE),
    # Times of day without a date component (e.g., "11.00 a.m.", "2:30 p.m.")
    re.compile(r'^\d{1,2}[.:]+\d{2}\s*(?:a\.?m\.?|p\.?m\.?)$', re.IGNORECASE),
    # Relative descriptions that are NOT specific dates
    re.compile(r'anniversary|expiry|exercise\s+period|business\s+days?\s+after', re.IGNORECASE),
    # Legal boilerplate date references
    re.compile(r'the\s+day\s+and\s+year\s+first\s+above', re.IGNORECASE),
    re.compile(r'day\s+and\s+year\s+first', re.IGNORECASE),
    # Descriptive date clauses (too long to be an actual date)
    re.compile(r'date\s+that\s+the\s+respective', re.IGNORECASE),
    re.compile(r'date\s+of\s+expiry', re.IGNORECASE),
    # "DATED this _____" style
    re.compile(r'^dated\s+this', re.IGNORECASE),
    # "this ____ day of" style
    re.compile(r'^this\s+.*day\s+of', re.IGNORECASE),
    # "no later than X days" — duration, not a date
    re.compile(r'no\s+later\s+than\s+\d+\s+days', re.IGNORECASE),
    # Complex temporal phrases that describe a calculation
    re.compile(r'upon\s+a\s+date\s+falling', re.IGNORECASE),
    # HTML tags (e.g., <sup>st</sup>)
    re.compile(r'<\s*sup\s*>', re.IGNORECASE),
]


def filter_false_positive_dates(dates):
    """Remove template placeholders and descriptive references from dates list."""
    filtered = []
    removed = []
    for date in dates:
        if any(p.search(date) for p in _DATE_BLACKLIST_PATTERNS):
            removed.append(date)
            continue
        filtered.append(date)
    if removed:
        print(f"[FILTER] Removed {len(removed)} false-positive dates: {removed}", flush=True)
    return filtered


# --- Email false-positive filter ---
def filter_false_positive_emails(emails):
    """Remove entries that don't contain an @ symbol (not real emails)."""
    filtered = []
    removed = []
    for email in emails:
        if '@' not in email:
            removed.append(email)
            continue
        filtered.append(email)
    if removed:
        print(f"[FILTER] Removed {len(removed)} false-positive emails: {removed}", flush=True)
    return filtered


# --- Bank account false-positive filter ---
_BANK_BLACKLIST_PATTERNS = [
    re.compile(r'\[.*?insert.*?\]', re.IGNORECASE),  # "[insert details of the bank account]"
    re.compile(r'\[.*?bank.*?\]', re.IGNORECASE),  # "[bank account number]"
    re.compile(r'^bank\s+account$', re.IGNORECASE),  # bare "bank account"
]


def filter_false_positive_bank_accounts(bank_accounts):
    """Remove template placeholders from bank accounts list."""
    filtered = []
    removed = []
    for ba in bank_accounts:
        if any(p.search(ba) for p in _BANK_BLACKLIST_PATTERNS):
            removed.append(ba)
            continue
        filtered.append(ba)
    if removed:
        print(f"[FILTER] Removed {len(removed)} false-positive bank_accounts: {removed}", flush=True)
    return filtered


# --- Registration ID false-positive filter ---
_REG_ID_BLACKLIST_PATTERNS = [
    re.compile(r'\[\s*-\s*\]', re.IGNORECASE),  # "[ - ]"
    re.compile(r'\[\s*_+\s*\]', re.IGNORECASE),  # "[____]"
]


def filter_false_positive_reg_ids(reg_ids):
    """Remove template placeholders from registration IDs list."""
    filtered = []
    removed = []
    for rid in reg_ids:
        if any(p.search(rid) for p in _REG_ID_BLACKLIST_PATTERNS):
            removed.append(rid)
            continue
        filtered.append(rid)
    if removed:
        print(f"[FILTER] Removed {len(removed)} false-positive reg_ids: {removed}", flush=True)
    return filtered


# ===============================
# REGEX BACKUP DETECTION
# ===============================
def regex_backup_detection(full_text, existing_reg_ids=None):
    """Catch common PII patterns the LLM might have missed using regex.

    Runs as a safety net after LLM extraction to ensure high-confidence
    patterns like emails, phone numbers, and IBANs are never missed.

    Args:
        full_text: The full document text.
        existing_reg_ids: List of already-detected registration IDs to exclude
                         from phone detection (prevents ИНН/ОГРН → phone confusion).
    """
    backup_emails = []
    backup_phones = []
    backup_ibans = []

    # Build set of digit-only versions of known registration IDs
    reg_id_digits = set()
    if existing_reg_ids:
        for rid in existing_reg_ids:
            digits = re.sub(r'\D', '', rid)
            if digits:
                reg_id_digits.add(digits)

    # Email pattern
    email_pattern = r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'
    for match in re.finditer(email_pattern, full_text):
        backup_emails.append(match.group())

    # International phone pattern (requires 7-15 digits)
    # Must have phone-like formatting: +, parentheses, or dashes
    phone_pattern = r'(?<!\d)(?:\+\d{1,3}[\s\-]?)?\(?\d{1,4}\)?[\s\-]?\d{2,4}[\s\-]?\d{3,4}(?!\d)'
    for match in re.finditer(phone_pattern, full_text):
        candidate = match.group().strip()
        digits = re.sub(r'\D', '', candidate)
        if 7 <= len(digits) <= 15:
            # Skip if this number matches a known registration ID
            if digits in reg_id_digits:
                continue
            # Skip if preceded by registration ID labels (ИНН, ОГРН, КПП, etc.)
            start_pos = match.start()
            prefix = full_text[max(0, start_pos - 20):start_pos]
            if re.search(r'(?:ИНН|ОГРН|КПП|ОКП|ОКПО|BIC|БИК|р/с|к/с|и/с)[:\s]*$', prefix, re.IGNORECASE):
                continue
            backup_phones.append(candidate)

    # IBAN pattern
    iban_pattern = r'\b[A-Z]{2}\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{0,4}\b'
    for match in re.finditer(iban_pattern, full_text):
        backup_ibans.append(match.group())

    return backup_emails, backup_phones, backup_ibans


# ===============================
# DEDUPLICATION
# ===============================
def dedup_list(items):
    """Deduplicate a list while preserving order (case-insensitive)."""
    seen = set()
    result = []
    for item in items:
        item = item.strip()
        if item and item.lower() not in seen:
            seen.add(item.lower())
            result.append(item)
    return result


def dedup_substrings(items):
    """Remove items that are substrings of other items.

    Optimized: uses a set for O(1) exact-match skips and limits
    comparison window for large lists.
    """
    if not items:
        return items
    sorted_items = sorted(items, key=len, reverse=True)
    result = []
    result_lower = []  # parallel list to avoid repeated .lower() calls
    for item in sorted_items:
        item_lower = item.lower()
        is_sub = False
        for accepted_lower in result_lower:
            if item_lower in accepted_lower:
                is_sub = True
                break
        if not is_sub:
            result.append(item)
            result_lower.append(item_lower)
    return result


# ===============================
# REPLACEMENT FUNCTIONS
# ===============================
def _flexible_pattern(text_str):
    """Build a regex that matches text_str with flexible whitespace,
    but NEVER matches in the middle of a word.

    Uses Unicode-aware word-character class so Cyrillic, Greek, and other
    non-Latin scripts are handled correctly (Python's \w with re.UNICODE).
    """
    escaped = re.escape(text_str)
    flexible = escaped.replace(r'\ ', r'\s+')
    # Use (?<!\w) and (?!\w) — with re.UNICODE flag these match Cyrillic/Greek too
    return r'(?<!\w)' + flexible + r'(?!\w)'


def build_combined_pattern(mapping):
    """Build a single compiled regex that matches all entities at once.

    This is dramatically faster than running one regex per entity,
    especially when there are hundreds of entities across many pages.
    Entities are sorted longest-first so longer matches take priority.
    """
    sorted_entities = sorted(mapping.keys(), key=len, reverse=True)
    patterns = []
    for entity in sorted_entities:
        patterns.append(_flexible_pattern(entity))
    combined = '|'.join(f'({p})' for p in patterns)
    return re.compile(combined, re.IGNORECASE | re.UNICODE), sorted_entities


def safe_replace(text, mapping):
    """Replace all entities in text using a single-pass combined regex.

    For small mapping sets (< 5), falls back to sequential replacement
    since the overhead of building a combined pattern isn't worth it.
    """
    if not mapping:
        return text

    if len(mapping) < 5:
        # Small number of entities — sequential is fine
        sorted_entities = sorted(mapping.keys(), key=len, reverse=True)
        for entity in sorted_entities:
            placeholder = mapping[entity]
            pattern = _flexible_pattern(entity)
            text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE | re.UNICODE)
        return text

    # For many entities, use single-pass replacement
    sorted_entities = sorted(mapping.keys(), key=len, reverse=True)
    patterns = [_flexible_pattern(entity) for entity in sorted_entities]
    combined = '|'.join(f'({p})' for p in patterns)

    try:
        compiled = re.compile(combined, re.IGNORECASE | re.UNICODE)
    except re.error:
        # Fallback to sequential if regex is too complex
        for entity in sorted_entities:
            placeholder = mapping[entity]
            pattern = _flexible_pattern(entity)
            text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE | re.UNICODE)
        return text

    # Build a lookup: for each match, find which entity it matched
    entity_lower_map = {}
    for entity in sorted_entities:
        entity_lower_map[entity.lower()] = mapping[entity]

    def replace_match(match):
        matched_text = match.group(0)
        # Look up the matched text (case-insensitive)
        matched_lower = matched_text.lower().strip()
        # Try exact match first
        if matched_lower in entity_lower_map:
            return entity_lower_map[matched_lower]
        # Normalize whitespace and try again
        normalized = re.sub(r'\s+', ' ', matched_lower)
        if normalized in entity_lower_map:
            return entity_lower_map[normalized]
        # Fallback: find the entity that best matches
        for entity, placeholder in mapping.items():
            if re.fullmatch(_flexible_pattern(entity), matched_text, re.IGNORECASE | re.UNICODE):
                return placeholder
        return matched_text  # no match — return unchanged

    text = compiled.sub(replace_match, text)
    return text


# ===============================
# MAIN ANONYMIZATION PIPELINE
# ===============================
def anonymize_document(pages, system_prompt=None, user_prompt=None):

    full_text = combine_pages(pages)
    total_tokens = len(tokenizer.encode(full_text, add_special_tokens=False))
    print(f"[LOG] Document: {len(pages)} pages, ~{total_tokens} tokens", flush=True)

    # Process page-by-page: each page becomes its own chunk(s)
    # This preserves natural document structure (headers, signature blocks, tables)
    chunks = pages_to_chunks(pages)
    print(f"[LOG] Split into {len(chunks)} chunks ({len(pages)} pages)", flush=True)

    if len(chunks) > MAX_CHUNKS:
        return {
            "error": (
                f"Document too large: {len(chunks)} chunks required but the maximum is {MAX_CHUNKS}. "
                "Please split the document and process it in parts."
            )
        }

    # Batch inference: process chunks in micro-batches via vLLM
    all_persons, all_orgs, all_dates, all_addresses, all_phones, all_reg_ids, all_bank_accounts, all_emails, all_passports, custom_entities = \
        extract_entities_batch(chunks, system_prompt=system_prompt, user_prompt=user_prompt)

    # ---- ANTI-HALLUCINATION: verify every entity exists in the source text ----
    full_text_lower = full_text.lower()
    all_persons = validate_entities(all_persons, full_text_lower, "persons")
    all_orgs = validate_entities(all_orgs, full_text_lower, "organizations")
    all_dates = validate_entities(all_dates, full_text_lower, "dates")
    all_addresses = validate_entities(all_addresses, full_text_lower, "addresses")
    all_phones = validate_entities(all_phones, full_text_lower, "phones")
    all_reg_ids = validate_entities(all_reg_ids, full_text_lower, "registration_ids")
    all_bank_accounts = validate_entities(all_bank_accounts, full_text_lower, "bank_accounts")
    all_emails = validate_entities(all_emails, full_text_lower, "emails")
    all_passports = validate_entities(all_passports, full_text_lower, "passports")
    custom_entities = validate_entities(custom_entities, full_text_lower, "custom_entities")

    # ---- FALSE-POSITIVE FILTERING: remove common misclassifications ----
    all_orgs = filter_false_positive_orgs(all_orgs)
    all_dates = filter_false_positive_dates(all_dates)
    all_emails = filter_false_positive_emails(all_emails)
    all_bank_accounts = filter_false_positive_bank_accounts(all_bank_accounts)
    all_reg_ids = filter_false_positive_reg_ids(all_reg_ids)

    print(f"[LOG] After false-positive filtering: {len(all_persons)} persons, {len(all_orgs)} orgs, "
          f"{len(all_dates)} dates, {len(all_addresses)} addresses, "
          f"{len(all_phones)} phones, {len(all_reg_ids)} reg_ids, "
          f"{len(all_bank_accounts)} bank_accounts, {len(all_emails)} emails, "
          f"{len(all_passports)} passports, {len(custom_entities)} custom_entities", flush=True)

    print(f"[LOG] After validation: {len(all_persons)} persons, {len(all_orgs)} orgs, "
          f"{len(all_dates)} dates, {len(all_addresses)} addresses, "
          f"{len(all_phones)} phones, {len(all_reg_ids)} reg_ids, "
          f"{len(all_bank_accounts)} bank_accounts, {len(all_emails)} emails, "
          f"{len(all_passports)} passports, {len(custom_entities)} custom_entities", flush=True)

    # ---- REGEX BACKUP: catch patterns the LLM might have missed ----
    backup_emails, backup_phones, backup_ibans = regex_backup_detection(
        full_text, existing_reg_ids=all_reg_ids
    )

    existing_emails_lower = {e.lower() for e in all_emails}
    for email in backup_emails:
        if email.lower() not in existing_emails_lower:
            all_emails.append(email)
            existing_emails_lower.add(email.lower())

    existing_phones_lower = {p.lower() for p in all_phones}
    for phone in backup_phones:
        if phone.lower() not in existing_phones_lower:
            all_phones.append(phone)
            existing_phones_lower.add(phone.lower())

    existing_ibans_lower = {b.lower() for b in all_bank_accounts}
    for iban in backup_ibans:
        if iban.lower() not in existing_ibans_lower:
            all_bank_accounts.append(iban)
            existing_ibans_lower.add(iban.lower())

    print(f"[LOG] After regex backup: {len(all_emails)} emails, "
          f"{len(all_phones)} phones, {len(all_bank_accounts)} bank_accounts", flush=True)

    # Deduplicate all entity lists
    unique_persons = dedup_substrings(dedup_list(all_persons))
    unique_orgs = dedup_substrings(dedup_list(all_orgs))
    unique_dates = dedup_substrings(dedup_list(all_dates))
    unique_addresses = dedup_substrings(dedup_list(all_addresses))
    unique_phones = dedup_substrings(dedup_list(all_phones))
    unique_reg_ids = dedup_substrings(dedup_list(all_reg_ids))
    unique_bank_accounts = dedup_substrings(dedup_list(all_bank_accounts))
    unique_emails = dedup_substrings(dedup_list(all_emails))
    unique_passports = dedup_substrings(dedup_list(all_passports))
    unique_custom = dedup_substrings(dedup_list(custom_entities))

    print(f"[LOG] After dedup: {len(unique_persons)} persons, {len(unique_orgs)} orgs, "
          f"{len(unique_dates)} dates, {len(unique_addresses)} addresses, "
          f"{len(unique_phones)} phones, {len(unique_reg_ids)} reg_ids, "
          f"{len(unique_bank_accounts)} bank_accounts, {len(unique_emails)} emails, "
          f"{len(unique_passports)} passports, {len(unique_custom)} custom_entities", flush=True)

    # Build mapping ordered by first appearance in the document
    def find_first_pos(token):
        pos = full_text.find(token)
        if pos == -1:
            pos = full_text.lower().find(token.lower())
        return pos if pos != -1 else float('inf')

    mapping = {}

    for i, org in enumerate(sorted(unique_orgs, key=find_first_pos), 1):
        mapping[org] = f"[COMPANY{i}]"

    for i, person in enumerate(sorted(unique_persons, key=find_first_pos), 1):
        mapping[person] = f"[PERSON{i}]"

    date_map = {}
    for i, d in enumerate(sorted(unique_dates, key=find_first_pos), 1):
        date_map[d] = f"[DATE{i}]"

    addr_map = {}
    for i, a in enumerate(sorted(unique_addresses, key=find_first_pos), 1):
        addr_map[a] = f"[ADDRESS{i}]"

    phone_map = {}
    for i, p in enumerate(sorted(unique_phones, key=find_first_pos), 1):
        phone_map[p] = f"[PHONE{i}]"

    reg_id_map = {}
    for i, r in enumerate(sorted(unique_reg_ids, key=find_first_pos), 1):
        reg_id_map[r] = f"[REG_ID{i}]"

    bank_account_map = {}
    for i, ba in enumerate(sorted(unique_bank_accounts, key=find_first_pos), 1):
        bank_account_map[ba] = f"[BANK_ACCOUNT{i}]"

    email_map = {}
    for i, e in enumerate(sorted(unique_emails, key=find_first_pos), 1):
        email_map[e] = f"[EMAIL{i}]"

    passport_map = {}
    for i, pass_num in enumerate(sorted(unique_passports, key=find_first_pos), 1):
        passport_map[pass_num] = f"[PASSPORT{i}]"

    custom_map = {}
    for i, c in enumerate(sorted(unique_custom, key=find_first_pos), 1):
        custom_map[c] = f"[ENTITY{i}]"

    # Combine all mappings for replacement
    all_mappings = {}
    all_mappings.update(mapping)
    all_mappings.update(addr_map)
    all_mappings.update(date_map)
    all_mappings.update(phone_map)
    all_mappings.update(reg_id_map)
    all_mappings.update(bank_account_map)
    all_mappings.update(email_map)
    all_mappings.update(passport_map)
    all_mappings.update(custom_map)

    print(f"[LOG] Total entities to replace: {len(all_mappings)}", flush=True)

    # Replace all entities page by page
    anonymized_pages = []
    for idx, page in enumerate(sorted(pages, key=lambda p: p["page"])):
        anon_text = safe_replace(page["text"], all_mappings)
        anonymized_pages.append({"page": page["page"], "text": anon_text})
        if (idx + 1) % 10 == 0:
            print(f"[LOG] Replaced entities in {idx + 1}/{len(pages)} pages", flush=True)

    print(f"[LOG] Anonymization complete for {len(pages)} pages", flush=True)

    display_mapping = dict(all_mappings)

    return {"pages": anonymized_pages, "mapping": display_mapping}


# ===============================
# RUNPOD HANDLER
# ===============================
def handler(event):
    try:
        pages = event["input"]["pages"]
        if not pages or not isinstance(pages, list):
            return {"error": "'pages' must be a non-empty list"}
        for p in pages:
            if "page" not in p or "text" not in p:
                return {"error": "Each page needs 'page' and 'text' fields"}

        system_prompt = event["input"].get("system_prompt", None)
        user_prompt = event["input"].get("user_prompt", None)

        print(f"[LOG] Received request with {len(pages)} pages (custom system prompt: {bool(system_prompt)}, custom user prompt: {bool(user_prompt)})", flush=True)
        return anonymize_document(pages, system_prompt=system_prompt, user_prompt=user_prompt)
    except KeyError as e:
        return {"error": f"Missing field: {e}"}
    except Exception as e:
        import traceback
        print(f"[ERROR] {traceback.format_exc()}", flush=True)
        return {"error": str(e)}


if __name__ == '__main__':
    # ===============================
    # LOAD MODEL WITH vLLM
    # (inside __main__ guard so vLLM's spawned child processes
    #  don't re-run initialization)
    # ===============================
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    llm = LLM(
        model=MODEL_PATH,
        dtype="float16",
        max_model_len=16384,        # long docs need more context room
        tensor_parallel_size=int(os.environ.get("TP_SIZE", "1")),
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,  # reuse KV cache for the shared system prompt
    )

    print(f"[LOG] Qwen3-8B loaded via vLLM (prefix caching ON)", flush=True)

    runpod.serverless.start({"handler": handler})