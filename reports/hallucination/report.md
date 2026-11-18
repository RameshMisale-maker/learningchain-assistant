# Hallucination evaluation

Labelled claims: 50 (assistant-authored; pending human review).
Positive class = supported. FP = an unsupported (hallucinated) claim the judge accepted.

## 1. Judge validation (confusion matrix vs. labelled claims)

| Judge | TP | FN | FP | TN | Accuracy | Precision | Recall | F1 | Specificity (hallucinations caught) | ms/claim |
|---|---|---|---|---|---|---|---|---|---|---|
| lexical | 25 | 3 | 5 | 17 | 0.840 | 0.833 | 0.893 | 0.862 | 0.773 | 0.2 |
| nli:nli-deberta-v3-small | 20 | 8 | 0 | 22 | 0.840 | 1.000 | 0.714 | 0.833 | 1.000 | 490.2 |
| nli:nli-deberta-v3-base | 23 | 5 | 0 | 22 | 0.900 | 1.000 | 0.821 | 0.902 | 1.000 | 813.1 |

**lexical** - hallucinations caught by type: wrong_entity 1/1, fabricated_feature 4/5, wrong_number 3/3, fabricated_policy 2/2, contradiction 2/6, unsupported_claim 1/1, unsupported_company_info 4/4
- C001 (supported, judged unsupported, score 0.0): LearningChain stores data on Microsoft Azure.
- C003 (supported, judged unsupported, score 0.0): LearningChain currently supports only PDF files for uploading certificates.
- C011 (supported, judged unsupported, score 0.0): LearningChain does not have access to or control over client data.
- C014 (unsupported, judged supported, score 0.5): DigiLocker stores certificates on a decentralized blockchain network.
- C031 (unsupported, judged supported, score 0.5): Registering a user requires the user's Aadhaar number.
- C034 (unsupported, judged supported, score 0.5): Certificate verification is free of charge for every verifier.
- C036 (unsupported, judged supported, score 0.6): Students can view all payment receipts from their dashboard.
- C050 (unsupported, judged supported, score 1.0): The June 2026 release removed the certificate verification page.

**nli:nli-deberta-v3-small** - hallucinations caught by type: wrong_entity 1/1, fabricated_feature 5/5, wrong_number 3/3, fabricated_policy 2/2, contradiction 6/6, unsupported_claim 1/1, unsupported_company_info 4/4
- C001 (supported, judged unsupported, score 0.016): LearningChain stores data on Microsoft Azure.
- C003 (supported, judged unsupported, score 0.0): LearningChain currently supports only PDF files for uploading certificates.
- C005 (supported, judged unsupported, score 0.002): Data uploaded to the blockchain through LearningChain stays valid for a lifetime.
- C011 (supported, judged unsupported, score 0.001): LearningChain does not have access to or control over client data.
- C013 (supported, judged unsupported, score 0.003): DigiLocker uses a centralized architecture for data storage and verification.
- C018 (supported, judged unsupported, score 0.002): Institution administrators upload scanned certificates in PDF format to the LearningChain portal.
- C021 (supported, judged unsupported, score 0.032): To revoke a certificate the admin opens the Reports module and then the Generated Certificates tab.
- C045 (supported, judged unsupported, score 0.001): You can email Sapphirus at info@sapphirus.in.

**nli:nli-deberta-v3-base** - hallucinations caught by type: wrong_entity 1/1, fabricated_feature 5/5, wrong_number 3/3, fabricated_policy 2/2, contradiction 6/6, unsupported_claim 1/1, unsupported_company_info 4/4
- C001 (supported, judged unsupported, score 0.001): LearningChain stores data on Microsoft Azure.
- C003 (supported, judged unsupported, score 0.0): LearningChain currently supports only PDF files for uploading certificates.
- C011 (supported, judged unsupported, score 0.005): LearningChain does not have access to or control over client data.
- C013 (supported, judged unsupported, score 0.144): DigiLocker uses a centralized architecture for data storage and verification.
- C027 (supported, judged unsupported, score 0.024): To reset a forgotten password click the Forget password option on the login page.

## 2. System answers

Judge: nli:nli-deberta-v3-base. Answer model: extractive (no LLM key).

Answers checked: 84, claims: 260, unsupported: 47 -> **hallucination rate 18.08%** (unsupported claims / total claims x 100). Fabricated URLs: 40. Invalid citations: 0.

| Category | Answers | Claims | Unsupported | Rate % | Fabricated URLs | Invalid citations |
|---|---|---|---|---|---|---|
| ambiguous | 1 | 11 | 0 | 0.0 | 2 | 0 |
| company | 17 | 30 | 6 | 20.0 | 3 | 0 |
| comparison | 2 | 5 | 2 | 40.0 | 0 | 0 |
| exact_detail | 5 | 30 | 8 | 26.67 | 8 | 0 |
| faq_exact | 18 | 31 | 3 | 9.68 | 0 | 0 |
| faq_paraphrase | 16 | 26 | 0 | 0.0 | 0 | 0 |
| misleading | 1 | 2 | 1 | 50.0 | 0 | 0 |
| multi_article | 2 | 13 | 3 | 23.08 | 2 | 0 |
| multi_part | 4 | 14 | 5 | 35.71 | 3 | 0 |
| products | 18 | 98 | 19 | 19.39 | 22 | 0 |

### Unsupported-claim examples (judge output - review manually)

- [C01] Give me a short introduction to Sapphirus. -> "Sapphirus Systems Pvt Ltd is a technology solutions company that turns emerging and enterprise technologies into practical solutions for real-world challenges" (score 0.001)
- [C01] Give me a short introduction to Sapphirus. -> "Blockchain is its core capability, complemented by Artificial Intelligence, Cloud, Data & Analytics and application engineering" (score 0.001)
- [C03] What's your email address? -> "You can reach Sapphirus by email at info@sapphirus.in, by phone at +91 96531 57225 or +91 86396 42312, or through the contact form at" (score 0.0)
- [C16] Where can I see current job openings? -> "Current openings at Sapphirus (full time, internship, part time and remote roles when available) are listed at" (score 0.003)
- [C18] Do you provide business intelligence dashboards? -> "Business Intelligence: reports, dashboards and performance views" (score 0.006)
- [P04] Why is a blockchain QR code more secure than an ordinary QR code? -> "Blockchain QR codes contain a unique hash value that is cross-checked with the blockchain when scanned" (score 0.042)
- [P04] Why is a blockchain QR code more secure than an ordinary QR code? -> "This ensures that the data is incorruptible and immutable" (score 0.267)
- [P05] How does an employer check a student's certificate? -> "Double-check the entered information to avoid errors" (score 0.171)
- [P05] How does an employer check a student's certificate? -> "Double-check all fields for correctness before generating the certificate" (score 0.003)
- [P06] Where do I find the verification link? -> "Where I can find the Verification URL?" (score 0.034)
- [P07] Steps to revoke a certificate? -> "Admin: How do I Revoke the Certificates" (score 0.002)
- [P10] How do I change my password in settings? -> "How do I Edit or Change My Password?" (score 0.108)
- [P10] How do I change my password in settings? -> "To Edit/Change my password, Click on Settings Module and then click on Change Password" (score 0.221)
- [P12] How do I generate a single certificate without a template? -> "How do I Secure the Student Digital Certificate without Template?" (score 0.011)
- [P12] How do I generate a single certificate without a template? -> "To secure the student digital certificate" (score 0.122)
