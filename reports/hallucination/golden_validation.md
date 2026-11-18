# Golden validation - https://learningchain.in/

- **golden_source**: https://learningchain.in/
- **questions**: 20
- **judge**: nli:nli-deberta-v3-base
- **answer_model**: openai/gpt-oss-120b
- **behaviour_accuracy**: 0.85
- **answer_accuracy_key_facts**: 0.694
- **negative_controls_declined**: 2/2
- **total_claims**: 43
- **unsupported_claims**: 16
- **hallucination_rate_pct**: 37.21
- **hallucination_rate_vs_golden_page_only_pct**: 79.07
- **fabricated_urls**: 0
- **seconds**: 359.7

| ID | Question | Type | Behaviour | Key facts | Claims | Unsupported |
|---|---|---|---|---|---|---|
| H01 | What is LearningChain? | golden_direct | ok | 1.0 | 1 | 0 |
| H02 | How does LearningChain work? | golden_direct | ok | 1.0 | 3 | 0 |
| H03 | What does an institution do in the issuance step? | refusal | WRONG | 0.0 | 0 | 0 |
| H04 | How is the QR code for a certificate created? | generated | ok | 1.0 | 6 | 5 |
| H05 | How can an employer verify a certificate? | generated | ok | 1.0 | 6 | 6 |
| H06 | Does LearningChain protect printed physical certificates? | generated | ok | 1.0 | 1 | 0 |
| H07 | What are the key features of LearningChain? | generated | ok | 0.5 | 5 | 0 |
| H08 | What are the benefits of blockchain for digital certificates? | generated | ok | 1.0 | 10 | 1 |
| H09 | Can a certificate secured with LearningChain be altered? | generated | ok | 1.0 | 1 | 0 |
| H10 | Can LearningChain certificates be verified from another country? | generated | ok | 1.0 | 1 | 0 |
| H11 | What kind of data storage does LearningChain use? | generated | ok | 1.0 | 1 | 1 |
| H12 | Who issues certificates with LearningChain? | refusal | WRONG | 0.0 | 0 | 0 |
| H13 | How does LearningChain help students share their achievements? | generated | ok | 1.0 | 2 | 1 |
| H14 | For how long are LearningChain credentials verifiable? | refusal | WRONG | 0.0 | 0 | 0 |
| H15 | How does adopting LearningChain help an institution? | generated | ok | 0.0 | 3 | 2 |
| H16 | Are LearningChain QR codes easy to use? | generated | ok | 1.0 | 1 | 0 |
| H17 | How quickly can a LearningChain certificate be verified? | generated | ok | 1.0 | 1 | 0 |
| H18 | What does LearningChain describe itself as? | golden_direct | ok | 0.0 | 1 | 0 |
| H19 | Does LearningChain mint certificates as NFTs on Ethereum? | refusal | ok | - | 0 | 0 |
| H20 | Does LearningChain offer a free mobile app for students on the Play Store? | refusal | ok | - | 0 | 0 |

## Unsupported claims (review manually - the judge is strict on paraphrases)

- H04: When configuring a certificate type, an admin must enable the QR‑code feature
- H04: In the certificate‑type settings, check “Enable Blockchain QR Code Generation” (with or without templates) 
- H04: Optionally, the admin can select “Logo in QR code” to embed a logo 
- H04: If the “Blockchain QR code generation without Templates” option is left unchecked, the QR code will be generated based on the selected template configuration 
- H04: Once enabled, the platform automatically generates the QR code and attaches it to the certificate (either embedded in the digital document or placed on a pre‑printed template)
- H05: Employers can verify a LearningChain certificate by using the platform’s verification tools
- H05: Log in to the LearningChain web application and go to the dashboard, where the Verification URL is displayed
- H05: Click the Verification URL to open the verification page and select the certificate type to start the 4‑step process
- H05: Enter the required details (student name, email, certificate type, certification name, issue month/year, academic reference number, plus verifier name, email, and mobile)
- H05: Complete the remaining steps as prompted (the guide continues beyond Step 2)
- H05: Alternatively, the certificate includes a unique QR code generated on the blockchain; scanning this QR code allows instant verification of the certificate’s authenticity
- H08: Builds trust and transparency among employers, institutions, and students
- H11: LearningChain uses decentralized storage on Microsoft Azure, with data kept across multiple nodes to provide secure, tamper‑proof storage
- H13: The platform’s blockchain‑secured credentials are always verifiable, giving students confidence when presenting their qualifications to employers or other institutions
- H15: Adopting LearningChain enables an institution to
- H15: Encourage students to share these secure credentials on social media, which enhances the institution’s visibility and expands its reach
