# LearningChain - Certificate and Blockchain QR Code Generation Modes

Product: LearningChain by Sapphirus - blockchain-powered academic credentialing platform.
Source: LearningChain chatbot FAQ (LearningChain_Chatbot_FAQ.csv), compiled from the Admin Training Manual, the product Overview presentation and the LearningChain Help Centre.

LearningChain can issue blockchain-secured certificates in ten ways. The mode depends on what the Admin enables in the configuration, what the user uploads, and where the blockchain QR code is placed.

## Summary of all certificate generation modes

| # | Mode | Admin configuration | What is uploaded | QR code placement | Typical use |
|---|---|---|---|---|---|
| 1 | Without template (digital only) | 'Blockchain QR code generation without templates' (default) | The digital document, while generating | QR code issued for the certificate | Digital certificates only |
| 2 | Without template + Embed QR | Without templates + 'Embed QR Code' (page number and position) | Certificate PDF (bulk: PDFs + Excel metadata) | Embedded at the chosen page and position | Final PDFs that need a QR |
| 3 | Pre-Printed (with templates) | 'With Templates' + 'Pre-Printed' | Pre-Printed Template PDF + Final Template PDF | Per template layout | Pre-printed stationery |
| 4 | Customer Supplied | 'With Templates' + 'Customer Supplied' | Final Template only | Selected position and page | Institution's own design |
| 5 | Pre-Printed + Customer Supplied | 'With Templates' + both checkboxes | Final Template PDF (bulk: JPG + Excel) | Set by coordinates | Original Degree style certificates |
| 6 | Pre-Printed + Photo field | Pre-printed + 'Photo' field enabled | Templates + student photo (bulk: JPG + Excel) | Per template | Certificates with a student photo (e.g. SSC) |
| 7 | Internal printing (no API) | 'Internal Printing' in the Certificates menu | Excel metadata, later the final PDF with QR codes | Embedded by the institution | COEP, DY Patil University |
| 8 | External printing (API) | API integration with the institution's platform | Data sent via API | Embedded by the integrated platform | Yashwantrao Chavan Maharashtra Open University |
| 9 | Old / already issued certificates | Blockchain QR for past certificates | Existing certificate data | QR for passed-out students' certificates | Securing legacy certificates |
| 10 | Direct digital certificates | Digital issuance | Certificate data | Delivered digitally to the student | Participation / appreciation certificates |

## Mode 1: How do I generate certificates without a template (digital only)?

- Admin configuration: enable 'Blockchain QR code generation without templates'. This is also the default when no option is enabled.
- What is uploaded: no template is needed. The user uploads the digital document while generating the certificate.
- QR code: a blockchain QR code is issued for the certificate.
- Who generates and how: the User, from Certificates > Generate Single (or Generate Bulk).
- Typical use: digital certificates only.
- Source: Admin Training Manual (slide 14); LearningChain Help Centre.

## Mode 2: How do I embed the blockchain QR code on my existing certificate PDFs (without template + Embed QR)?

- Admin configuration: enable 'Blockchain QR code generation without templates' together with 'Embed QR Code', then choose the page number and the QR position.
- What is uploaded: the certificate PDF. For bulk generation, upload the PDFs plus an Excel metadata file.
- QR code: embedded automatically at the chosen page and position.
- Who generates and how: the User, from Generate Single or Generate Bulk.
- Typical use: you already have final PDF certificates and need a blockchain QR code on them.
- Source: Admin Training Manual (slide 15); LearningChain Help Centre.

## Mode 3: How do I generate certificates on pre-printed stationery (Pre-Printed templates)?

- Admin configuration: enable 'Blockchain QR Code Generation with Templates' and select 'Pre-Printed'.
- What is uploaded: a Pre-Printed Template (PDF) and a Final Template (PDF).
- QR code: placed according to the template layout.
- Who generates and how: the User, from Generate Single; bulk generation uses Excel metadata.
- Typical use: printing on pre-printed stationery, for example a Provisional Certificate.
- Source: Admin Training Manual (slide 16); LearningChain Help Centre.

## Mode 4: How do I use our own certificate design (Customer Supplied template)?

- Admin configuration: enable 'With Templates' and select 'Customer Supplied'; select the QR position and the page number.
- What is uploaded: the Final Template only.
- QR code: placed at the selected position and page.
- Who generates and how: the User, from Generate Single; the Student Review Process is optional.
- Typical use: the institution provides its own certificate design.
- Source: Admin Training Manual (slides 17-18).

## Mode 5: How do I generate Original Degree style certificates (Pre-Printed + Customer Supplied)?

- Admin configuration: enable 'With Templates' and tick both 'Pre-Printed' and 'Customer Supplied'; the QR position is set by coordinates.
- What is uploaded: the Final Template (PDF). For bulk generation, upload JPG images plus an Excel metadata file.
- QR code: placed by coordinates - verify its size and position before generating.
- Who generates and how: the User, from Generate Single or Generate Bulk. The Admin can enable 'With Single Approval'.
- Typical use: Original Degree style certificates.
- Source: LearningChain Help Centre.

## Mode 6: How do I issue certificates with a student photo (Pre-Printed + Photo field)?

- Admin configuration: Pre-printed set to True and the 'Photo' field enabled.
- What is uploaded: the templates plus the student photo, which must meet the required dimensions and resolution. For bulk generation, upload JPG images plus an Excel metadata file.
- QR code: placed according to the template.
- Who generates and how: the User, from Generate Single or Generate Bulk.
- Typical use: certificates that carry the student's photo, for example SSC certificates.
- Source: LearningChain Help Centre.

## Mode 7: How does internal printing work (printing at the institution, no API)?

- Admin configuration: use 'Internal Printing' in the Certificates menu.
- What is uploaded: Excel metadata first; later, the final PDF with the QR codes embedded by the institution.
- QR code: the institution embeds the downloaded QR codes itself.
- Who generates and how: the User generates the QR codes, downloads them from Printer Generated Certificates, prints / embeds them, then uploads the final PDF.
- Typical use / examples: COEP, DY Patil University.
- Source: Admin Training Manual (slides 47-56); Overview presentation (slides 11-12).

## Mode 8: How does external printing through an API work?

- Admin configuration: API integration with the institution's own platform.
- What is supplied: the data is sent via the API from the institution's system.
- QR code: the blockchain QR code is embedded by the integrated platform.
- Who generates and how: the institution's own platform, through the API.
- Typical use / example: Yashwantrao Chavan Maharashtra Open University.
- Source: Overview presentation (slides 9-10).

## Mode 9: Can LearningChain secure old or already issued certificates?

- Yes. A blockchain QR code can be generated for past certificates from the existing certificate data.
- QR code: issued for passed-out students' certificates.
- Typical use: securing legacy certificates that were issued before the institution started using LearningChain.
- Source: Overview presentation (slide 14).

## Mode 10: Can LearningChain issue direct digital certificates?

- Yes. Certificates can be issued digitally from the certificate data and delivered digitally to the student.
- Typical use: participation and appreciation certificates.
- Source: Overview presentation (slide 13).
