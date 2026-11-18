# LearningChain Help Centre - General

Product: LearningChain by Sapphirus - blockchain-powered academic credentialing platform.
Source: LearningChain Help Centre https://learningchainai.zendesk.com/hc/en-gb

Frequently asked questions and product release notes (what's new in each release).

## Release notes: 02-June-2026 Release notes

Article: https://learningchainai.zendesk.com/hc/en-gb/articles/59593695398425-02-June-2026-Release-notes

Updated: 2026-07-02

User Story ID: 12660

Title: Add Verification Manual Link/PDF on Certificate Verification Page

Business Requirement:

The Certificate Verification page should provide users with easy access to a Verification Instruction Manual that explains the complete certificate verification process. This will help users understand the verification steps, required inputs, expected outcomes, and common troubleshooting guidance.

Administrators should be able to upload, update, or remove the instruction manual through the Admin module, and the changes should be reflected immediately on the Online Verification page.

Development change:

A new Verification Manual feature has been implemented to improve the certificate verification experience.

The following enhancements have been made:

- Added a Verification Instruction Manual link on the Online Verification page.

- Users can click the link to open the instruction manual in a new browser tab without interrupting the verification process.

- Added an Admin configuration to upload and manage the verification manual document.

- Administrators can upload a new instruction manual (PDF).

- Administrators can replace the existing manual with an updated version.

- Administrators can remove the uploaded manual when it is no longer required.

- Any changes made by the administrator are reflected immediately on the Online Verification page.

User Story ID: 13804

Title: Registration – Region-Based Mobile Input & Edit User Profile

Business Requirement:

The system should support region-based mobile number input during user registration and profile management. Users should be able to select their country/region, and the corresponding country code should be applied to the mobile number field.

Additionally, users should be able to edit their profile information after registration to keep their personal details up to date.

Development change:

The Registration and Edit User Profile modules have been enhanced to support region-based mobile number input and profile updates.

The following enhancements have been implemented:

- Added Country/Region selection for the mobile number field during user registration.

- Mobile number field automatically displays the appropriate country dialing code based on the selected region.

- Added validation to ensure mobile numbers are entered in the correct format for the selected region.

- Enhanced the Edit User Profile screen to allow users to update their personal information, including their mobile number with the selected country code.

- Updated profile information is saved and reflected immediately in the system.

User Story ID: 13806

Title: As a SysAdmin, I want to customize Region-Based Mobile Number Formatting Configuration in Certificate Verification.

Business Requirement:

The system should support region-based mobile number formatting in the Certificate Verification module. Mobile numbers displayed during the verification process should follow the configured country/region format, ensuring consistency, accuracy, and improved readability for users across different regions.

Development change:

The Certificate Verification module has been enhanced to support region-based mobile number formatting.

The following enhancements have been implemented:

- Added support for configurable region-based mobile number formatting in the Certificate Verification module.

- Mobile numbers are displayed with the appropriate country/region dialing code based on the configured region.

- Applied consistent mobile number formatting throughout the certificate verification process.

- Ensured the configured format is reflected wherever student mobile numbers are displayed in the verification workflow.

- Maintained compatibility with existing certificate verification functionality.

User Story ID: 13807

Title: As a SysAdmin, I want to customize Region-Based Mobile Number Formatting Configuration in Bulk Upload

Business Requirement:

The system should support region-based mobile number formatting during the Bulk Upload process. Mobile numbers uploaded through bulk import should be validated and formatted according to the configured country/region settings, ensuring consistency and accuracy of student data across the application.

Development change:

The Bulk Upload module has been enhanced to support region-based mobile number formatting.

The following enhancements have been implemented:

- Added support for configurable region-based mobile number formatting during bulk upload.

- Mobile numbers are validated based on the configured country/region format.

- The appropriate country/region dialing code is applied where required.

- Invalid mobile numbers that do not match the configured regional format are identified during the upload process.

- Ensured that all successfully uploaded mobile numbers are stored in a standardized format across the application.

User Story ID: 13809

Title: As a SysAdmin, I want to customize Region-Based Mobile Number Formatting Configuration in Print/Save Student Details and View Marks List.

Business Requirement:

The system should support region-based mobile number formatting when displaying, printing, or exporting student information. The formatting should be configurable so that mobile numbers are presented according to the selected country/region standard, ensuring consistency across all student-related reports and screens.

Development change:

The Print/Save Student Details and View Marks List modules have been enhanced to support region-based mobile number formatting.

The following enhancements have been implemented:

- Added support for region-based mobile number formatting.

- Mobile numbers are displayed with the appropriate country/region code based on the configured settings.

- Applied the configured format in the Print Student Details functionality.

- Applied the configured format in the Save/Export Student Details functionality.

- Applied the configured format in the View Marks List screen.

- Ensured consistent mobile number formatting across all supported modules.

User Story ID: 14839

Title: Incorrect PDF Formatting in Verification Summary (Extra Blank Page & Stamp Misalignment)

Business Requirement:

The Verification Summary PDF generated after the Online Verification process should be formatted correctly as a single-page document. The document should not contain any extra blank pages, and the institution's official stamp should be properly aligned and fully visible. This ensures that the verification summary is accurate, professional, and suitable for official use.

Development change:

The PDF generation logic for the Verification Summary has been corrected to resolve formatting issues.

The following enhancements have been implemented:

- Fixed the issue where the Verification Summary PDF was generated with an extra blank page.

- Corrected the layout to ensure the PDF is generated as a single-page document.

- Fixed the alignment of the YCMOU official stamp so that it is displayed completely within the same page.

- Improved the PDF rendering to maintain a consistent layout across all verification cases.

- Ensured that the Verification Summary PDF attached to verification emails is generated correctly without formatting issues.

## Release notes: 24-March-2026 - Release Notes

Article: https://learningchainai.zendesk.com/hc/en-gb/articles/59593446111001-24-March-2026-Release-Notes

Updated: 2026-07-02

User Story ID: 13311

Title: As a Student, I want to review my certificate details before approval so that incorrect data can be corrected by sending review emails to students.

Description: Introduced a Student Review workflow that allows students to verify their certificate details before final approval and generation. When Student Review is enabled for a certificate type, the system sends review emails to students containing a secure link to review their certificate information.

Business Requirement:

To provide students with an opportunity to validate their personal and academic details before certificate generation, reducing errors in the certificate issuance process.

Functional Changes:

Added a configurable Student Review workflow for certificate generation.

When Student Review is enabled, certificate records are displayed in the Student Review module based on the selected batch.

Administrators can select one or more records and trigger Bulk Email notifications to students.

Each student receives an email containing:

- Certificate review notification.

- Secure review link.

-  Instructions to approve or request corrections.

After the email is sent, the Student Review Status is automatically updated to Pending.

Students can review their certificate details and choose to:

-  Approve the details if they are correct.

-  Request Correction if any information needs to be updated.

Once the student approves the certificate, the status changes to Reviewed.

If the student requests corrections, the status changes to Correction Request.

Certificates do not proceed to the next stage until the student completes the review process or the configured review period expires.

Attaching the screenshots for reference.

User Story ID: 13314

Title: As an University_Admin, I want to upload a certificate background so that certificates can be dynamically generated.

Description:

Introduced a new configuration in the Certificate Type screen that allows University Administrators to upload a certificate background image. During certificate generation, the system dynamically overlays student details and the QR code onto the uploaded background to create the final certificate.

Business Requirement:

To provide flexibility for universities that use certificate background designs or pre-printed stationery instead of fully designed templates, enabling dynamic certificate generation with student data and QR codes.

Functional Changes:

- Added an option to upload a certificate background in the Certificate Type configuration.

- Supports certificate generation using background images instead of fully designed templates.

- System automatically places student information and QR code on the uploaded background during certificate generation.

- Enables dynamic generation of certificates based on the configured background.

User Story ID: 13320

Title: As a University Administrator, I want to define the exact QR code position on the certificate so that the QR code appears correctly on every generated certificate.

Feature Enhancement

QR Code Position Configuration with Drag-and-Drop Interface

Description:

A new QR Code Position Configuration feature has been implemented within the Certificate Type configuration.

This enhancement allows the University Administrator to upload a pre-printed certificate template and visually configure the exact placement and size of the QR code using an intuitive drag-and-drop interface.

The configured X and Y coordinates, along with the QR code dimensions (width and height), are saved as part of the certificate type configuration. During certificate generation, the system automatically places the QR code at the configured location with the defined size, ensuring consistent positioning across all generated certificates.

Functional Changes:

- Added Drag-and-Drop QR Placeholder functionality.

- Upload and preview the certificate template before configuring the QR position.

- Configure the exact X & Y coordinates of the QR code.

- Configure the QR code size (Width & Height).

- Save QR position and size as part of the Certificate Type configuration.

- Automatically apply the saved configuration during certificate generation.

- Eliminates manual positioning and ensures consistent QR placement across all certificates

Business Requirement:

The University requires the ability to configure the exact position and size of the QR code on certificate templates. Since different certificate layouts have different designs and reserved spaces, a fixed QR code position does not meet all requirements. The system should provide a configurable interface that allows the University Administrator to define the QR code placement and dimensions for each certificate type, ensuring accurate and consistent QR code positioning on all generated certificates.

Development change:

A new QR Code Position Configuration feature has been implemented in the Certificate Type module. The enhancement provides a drag-and-drop interface that enables the University Administrator to upload a certificate template, visually position the QR placeholder, and configure the QR code size. The system captures and saves the configured X & Y coordinates along with the QR code dimensions (Width & Height) as part of the certificate type configuration. During certificate generation, the QR code is automatically rendered at the configured position and size, ensuring consistent placement across all generated certificates while eliminating manual adjustments and alignment issues.

User Story ID: 13338

Title: As an Institution Admin, I want to approve certificates so that valid certificates are issued.

Description:

The system should support bulk certificate approval and processing to efficiently handle large volumes of certificate generation. Institution Administrators should be able to initiate bulk certificate approval and monitor the processing status in real time. The solution should support parallel processing, improve performance, and provide visibility into the progress of each batch.

Business Requirement:

A new Bulk Certificate Processing mechanism has been implemented to improve the performance and monitoring of certificate generation.

The following enhancements have been made:

- Institution Administrators can initiate bulk certificate approval.

- Once approved, the uploaded batch is automatically processed in the background.

- Each student certificate is processed independently using Azure Queue and Azure Functions, enabling parallel execution.

- Certificate generation status is updated for each student as Pending, Completed, or Failed.

- The system dynamically calculates the overall batch progress based on individual student processing status.

- A new Batch Progress window has been added to the Admin Dashboard to monitor certificate generation in real time.

Functional Changes:

- Enables efficient processing of large certificate batches.

- Provides real-time visibility into certificate generation progress.

- Improves system performance through parallel processing.

- Reduces manual monitoring effort for administrators.

- Ensures reliable and scalable certificate generation.

User Story ID: 13010

Title: NICMAR & LJK: Multi-Page PDF Split, Excel Mapping, and Azure Blob Upload

Description:

Introduced a new bulk processing feature that allows users to upload an Excel file and a combined PDF document through the application. The system automatically generates a Batch ID, stores the uploaded files in Azure Blob Storage, and processes them asynchronously.

Business Requirement:

To provide an automated and scalable solution for processing large volumes of PDF documents and Excel records in bulk. The system should enable users to upload a combined PDF and an Excel file, automatically split the PDF into individual documents, map each document with the corresponding record using PRN/Seat Number, and securely store the processed files in Azure Blob Storage.

This enhancement eliminates manual document splitting and mapping efforts, improves processing efficiency, supports large-volume uploads, and ensures reliable tracking of each processing request through a unique Batch ID.

Functional Changes:

- Added support for uploading a combined PDF and Excel file in a single batch.

- System automatically generates a unique Batch ID for tracking each upload.

- Large multi-page PDFs are automatically split into individual PDF files.

- Extracted PRN/Seat Number from each PDF and mapped it with the corresponding Excel record.

- Generated individual PDF file names based on the extracted data.

- Updated the uploaded Excel file with generated PDF names and processing details.

- Uploaded all generated PDFs and the updated Excel file to Azure Blob Storage.

- Implemented Azure Function-based background processing for improved performance and scalability.

User Story ID: 13316

Title: As an University_Admin, I want to preview certificates before generation so that layout issues can be detected.

Description: Introduced a certificate preview feature that allows University Administrators to verify the certificate layout before generation or sending review emails. The preview displayed is determined based on the Certificate Type configuration.

Business Requirement:

To enable administrators to validate the certificate appearance before generation, helping to identify layout, formatting, or QR code placement issues.

Functional Changes:

Added support for certificate preview based on Certificate Type Configuration.

Introduced two preview options:

- Pre-Printed Template Preview

- Final Template Preview

The system automatically displays the configured preview type.

Certificate preview is available in the Admin Approval screen.

The same preview is also available to students through the Student Review email link.

Administrators and students can verify the certificate layout, student data, and QR code placement before proceeding.

User Story ID: 13475

Title: As a University Admin, I want to configure certificate approval settings in the Certificate Type screen, so that each certificate type can have its own approval workflow and support the Student Review process.

Business Requirement:

Previously, the approval settings for certificate generation were configured globally in the Settings screen. This approach applied the same approval workflow to all certificate types.

The system should allow approval settings to be configured at the Certificate Type level, enabling each certificate type to have its own approval workflow. This provides greater flexibility and ensures that the Student Review process and certificate generation follow the approval configuration defined for the selected certificate type.

Development change

The approval configuration has been moved from the Settings screen to the Certificate Type screen.

The following enhancements have been implemented:

- Added Single Certificate Generation Approval option in the Certificate Type configuration.

- Added Bulk Certificate Generation Approval option in the Certificate Type configuration.

- Administrators can configure approval settings while creating or editing a certificate type.

- Each certificate type can have its own approval workflow.

- The Student Review process now follows the approval configuration defined for the respective certificate type.

- All certificate generation processes (Single and Bulk) now use the approval settings configured in the Certificate Type instead of the global Settings screen.

User Story ID: 13498

Title: As an admin, I want to create and edit student review mail templates and view them categorized by certificate type in the mail templates screen.

Description: The system should allow administrators to create and manage Student Review email templates for different certificate types. Each certificate type may require a different email format and content during the Student Review process.

Administrators should be able to create, edit, and manage mail templates, associate them with a specific certificate type, and easily view or filter templates based on the selected certificate type.

Business Requirement:

The Mail Templates module has been enhanced to support Student Review mail templates.

The following enhancements have been implemented:

- Added the ability to create Student Review mail templates.

- Added support for Rich Text Editor to compose the email body.

- Added Certificate Type selection while creating a mail template.

- Added Placeholder selection to insert dynamic values into the email template.

- Mail templates are saved and linked to the selected certificate type.

- Administrators can edit existing mail templates, and the updated content is reflected immediately after saving.

- Enhanced the Mail Templates List screen to display templates by Certificate Type.

- Added filtering support to quickly view templates for a specific certificate type.

Functional Changes:

- Enables certificate type-specific email communication.

- Simplifies management of Student Review email templates.

- Improves consistency of emails sent during the Student Review process.

- Reduces manual effort by using reusable templates with dynamic placeholders.

- Makes it easier for administrators to locate and manage templates through certificate type-based filtering.

User Story ID: 13542

Title: Append Back Page to Final Certificate PDF Using Syncfusion Based on Certificate Name Configuration

Business Requirement:

The system should support appending a back page to the generated certificate PDF based on the configuration defined for the selected Certificate Name.

Different certificate names may require different back pages. During certificate generation, the system should automatically identify the configured back page and append it to the final certificate PDF without affecting the existing certificate layout or content.

Development change:

The certificate generation process has been enhanced to support automatic back page merging using Syncfusion PDF.

The following enhancements have been implemented:

- The system checks whether a back page is configured for the selected Certificate Name.

- If a back page is available, it is automatically appended as the last page of the generated certificate PDF.

- The back page is merged using Syncfusion PDF while preserving its original formatting and layout.

- Different certificate names can have different back page configurations.

- If no back page is configured, the system generates the certificate as a single-page PDF without any changes.

User Story ID: 10803

Title: Send Student Review Notification via Email to Students Based on Batch Number

Description: Introduced a bulk email notification feature that enables administrators to send student review emails for a selected batch. The system automatically retrieves student records associated with the “Batch ID” and sends review notifications containing document and review links, allowing students to access and review their documents.

Business Requirement: To automate the student review notification process for bulk-generated certificates/documents, ensuring timely communication with students while supporting large-scale email delivery.

Development Change: Added a bulk email notification feature that allows administrators to send student review emails based on Batch ID. The system sends review links to students, tracks email delivery status, and supports scalable queue-based processing with retry mechanisms for failed emails.

Functional Changes:

- Added Bulk Email Sending functionality from the Bulk Approvals Grid.

- Administrators can trigger email notifications for all students within a selected batch.

- System retrieves student details and document links based on the selected Batch ID.

- Review emails are automatically sent to students with document and review links.

- Implemented Azure Queue-based processing to handle large volumes of email notifications efficiently.

- Added asynchronous email delivery using Azure Functions and SendGrid integration.

- Introduced email tracking to monitor delivery status and email engagement.

- Captures email events such as Delivered, Opened, Bounced, and Dropped.

- Added support for tracking email status against each student and batch.

- Implemented retry mechanism for failed email delivery attempts.

User Story ID: 12659

Title: Enable Preview of Final Certificate Template Based on Certificate Type Configuration

Description: Enhanced the certificate preview functionality to display either the Pre-Printed Template or the Final Certificate Template based on the Certificate Type Configuration. This allows users to verify the actual certificate appearance before generation.

Business Requirement:

To provide users with an accurate preview of the certificate before generation, ensuring that the displayed preview matches the actual certificate output that will be generated.

Functional Changes:

Certificate Type Configuration:-

A new configuration has been introduced under the Template Configuration section to control the certificate preview behaviour.

Preview Template Selection Options

The system now provides the following preview options:

-  Use Pre-Printed Template for Preview

  - The system displays the pre-printed certificate template as the preview.

-  Use Final Template or Final Background for Preview

  - The system displays the final certificate output as the preview.

  - The preview includes all template configurations and dynamic certificate elements that will appear on the generated certificate.

System Behaviour

- The preview displayed to the user is determined based on the configuration selected in the Certificate Type settings.

- The system automatically identifies the configured preview type and renders the corresponding certificate preview.

- Users can validate the final certificate appearance before initiating certificate generation.

User Flow

Step 1: Login to the LearningChain Admin Portal.

Step 2: Navigate to Certificates → Add Certificate Type.

Step 3: Configure the certificate type under the Template Configuration section.

Step 4: Select the required template generation method:

  - Blockchain QR Code Generation with Templates.

Step 5: Under Preview Template Selection, choose one of the following:

  - Use Pre-Printed Template for Preview

  - Use Final Template or Final Background for Preview

Step 6: Save the Certificate Type configuration.

Step 7: During certificate preview, the system will display the preview based on the selected configuration.

Step 8: Verify the certificate layout and content before proceeding with certificate generation.

User Story ID: 12866

Title: Direct PDF View in Student Review Link with Review & Edit Options

Description: Enhanced the Student Review process to provide a seamless certificate review experience. Previously, students were required to open the review link, navigate to the certificate details page, and click the Preview icon to view the certificate PDF. With this enhancement, the certificate PDF is displayed directly when the student accesses the review link from the email.

Business Requirement:

The Student Review workflow to display the certificate PDF directly from the email review link. Review and Edit options are available alongside the PDF.

Functional Changes:

- Certificate PDF now opens automatically when the student clicks the review link received via email.

- Eliminated the need for an additional Preview button click to view the certificate.

- Review and Edit options are displayed alongside the certificate PDF for easy access.

- Removed the watermark from the Specimen Copy/Certificate PDF during the review process.

- Students can now view a certificate preview before submitting their review.

User Story ID: 13310

Title: As an System administrator, want to enable or disable student review for certificate generation so that for different University Clients have customized workflow.

Description: Introduced a configurable Student Review Required option in the Certificate Type configuration. This allows administrators to enable or disable the student review process based on the university's certificate generation workflow requirements.

Business Requirement:

To provide universities with the customize their certificate generation workflow by enabling or disabling student review before certificate issuance.

Development Change: Added a Student Review Required configuration in Certificate Type settings. When enabled, students must review and approve certificate details before generation. When disabled, certificates are generated directly without the student review process.

Functional Changes:

- Added a Student Review Required option in the Certificate Type configuration.

- When enabled, students must review and approve their certificate details before certificate generation.

- The student review workflow is applicable to both Single Certificate Generation and Bulk Certificate Generation.

- Review notifications are sent to students when the review process is enabled.

- When disabled, certificates are generated directly without requiring student approval.

- Allows different university clients to configure certificate workflows based on their business requirements.

User Story ID: 13331

Title: As an Admin, want to generate certificates in bulk using Excel so that large batches can be created. (Bulk Certificate Generation Using Excel Upload) With regards to student review process.

Description: Introduced a bulk certificate generation feature that allows University Administrators to generate certificates for multiple students by uploading an Excel file containing student data along with the certificate (PDF) template. The system processes each record and creates certificates in bulk.

Business Requirement:

To enable university administrators to generate certificates for large batches of students through a single Excel upload process along with the PDF Template, minimizing manual certificate creations.

Development change: Added support for bulk certificate generation through Excel upload. The system automatically generates certificates for multiple students and supports configurable workflows, including Individual/Combined output, Approval, and Student Review options. When Student Review is enabled, review emails are sent automatically to students.

Functional Changes:

- Added support for bulk certificate generation through Excel file upload.

- System processes each student record from the uploaded Excel file and generates certificates automatically.

- Generated certificates are saved with "Saved" status for further processing.

- Supports both Individual and Combined certificate output formats.

- Supports workflows with or without Approval.

- Supports workflows with or without Student Review.

- When Student Review is enabled, review emails are automatically sent to students, and the review status is marked as Pending.

- When Student Review is disabled, certificates directly proceed to the approval workflow without sending review emails.

User Story ID: 13327

Title: As a University Administrator, I want to configure certificate back pages year-wise so that multiple versions can be maintained (Optional).

Business Requirement:

Universities may update the certificate back page content across different academic years due to changes in regulations, policies, branding, or certificate formats. The system should support maintaining multiple versions of certificate configurations by academic year, allowing administrators to easily manage and retrieve the appropriate certificate configuration.

To simplify administration, the Certificate Name module should provide a Year-wise filter, enabling administrators to quickly locate certificate configurations for a specific academic year. This reduces the effort required to manage a growing number of certificate templates and improves the overall user experience.

Development change:

The Certificate Name module has been enhanced to support year-wise certificate configuration management.

The following changes have been implemented:

- Added a Year field while creating a new Certificate Name.

- Introduced a Year-wise filter on the Certificate Name listing page.

- Administrators can filter certificate configurations based on the selected academic year.

- The Year filter can be used along with existing search functionality to quickly locate the required certificate configuration.

- This enhancement enables universities to maintain multiple certificate back page versions for different academic years without affecting existing configurations.

## Release notes: July 11th, 2024 Release Notes

Article: https://learningchainai.zendesk.com/hc/en-gb/articles/35420870136729-July-11th-2024-Release-Notes

Updated: 2024-07-23

Bugs Fixes:

-  Fixed: Issue with QR-Code File Name Convention, Now the “QR-Code file name” convention is working as excepted.

- Fixed: Request to Modify CGPA and SGPA Input Data Validation- Decimal, now it was allowing “Two decimal” values with “0 to 10” as its value.

- Fixed: Print button on the Generation and verification summary, Now the “Institution logo” is displayed on the print page as expected.

New features

- New: Added New extra custom fields in certificate type like “CustomField8, CustomField9, CustomField10, Barcode, Certification Code”.

## Release notes: April 19, 2024 Release Notes

Article: https://learningchainai.zendesk.com/hc/en-gb/articles/32444053532441-April-19-2024-Release-Notes

Updated: 2024-05-14

Bug Fixes

-  Fixed: Unable to download bulk QR codes through Action Dropdown in Verified Certificates

It is now functioning as expected. Now we can download QR codes in bulk.

After Clicking on Go all the selected student’s “QR Codes” will be downloaded automatically.

-  Fixed: Issue with Missing Download options of original certificate and QR Code options, In the certificate generation e-mail

Now original certificate and QR Code options are visible in the mail and this functioning as expected.

-  Fixed: Empty Fields should not displayed on the Visual Verification page

We have added a new feature: empty fields will not displayed on the Visual Verification page.

-  Fixed: Certificate Name Editing Issue with Slash "/" symbol in the name
Now special characters are accepted as “Certificate Name” in Bulk Generation Certificates, It was working as expected.

New Features

New:

-  Now we have the New features in the Reports in the dashboard in this we can drill down the individual Student Wallet Report. The reports contain all the info, including counts for wallets and the generated certificate details.

## FAQ: Digital Certificates (Digi Locker) Vs BlockChain Based Certificates

Article: https://learningchainai.zendesk.com/hc/en-gb/articles/33328645460889-Digital-Certificates-Digi-Locker-Vs-BlockChain-Based-Certificates

Updated: 2024-06-04

Digital Certificates

- Digital Certificates are stored in a centralized database, prone to failure.

- Digital Certificates without digital signatures are extremely easy to forge

- Registries of digital certificates are prone to large scale data leaks

- Keeping digital records safe requires sophisticated and multi-tier backup systems which are prone to failure

- The use of digital signatures requires the participation of third-party providers that guarantee the integrity of the transaction; they exert control on every aspect of the certification and verification processes, which can lead to abuses.

- There is no universally accepted open standard for digital signatures, leading to certificates that can only be verified within the context of proprietary software.

- It is easy to erase electronic records unless very sophisticated security and backup systems are established.

- If the digital record fails, the certificate loses its value because of the lack of intrinsic value.

- Digital certificates are liable to be downloaded in bulk.

- Maintaining IT infrastructure for Digital Certificates is very expensive and ever-increasing.

- Verifying the digital records are not instant, takes considerable time up to 24 hours, and may increase.

- Verifying the digital records requires authorization from the stack holder(s), prone to failures, less reliable.

- Verifying the digital records may not be possible if issuing organization is no more available or stack holders not willing.

Blockchain Certificates

- Blockchain Certificates are stored in a decentralized database over the cloud infrastructure.

- Impossible to change the records (immutable) as not stored in a single place.

- Validation and Verification are possible even if the issuing organization is no longer exists.

- Verification is very easy, anyone having access to Blockchain can verify the certificates.

- They are verified instantly, using a QR Code

- Option not to publish the certificate itself, preserving its privacy

- Each certificate will contain a unique digital signature (Hash, which can't be tampered with and is unique).

- Which is stored in a public database (the blockchain) and identical in thousands of computers around the world.

- The digital signature is linked with a blockchain address. In any case, the original document is tampered with and regenerated, the certificate fails during the verification process.

- No chance of forging.

- The registry of certificates issued and received in a blockchain can only be destroyed if each and every one of the existing copies are destroyed in thousands of computers that host this technology.

## Release notes: March 06, 2024 Release Notes

Article: https://learningchainai.zendesk.com/hc/en-gb/articles/32445419020185-March-06-2024-Release-Notes

Updated: 2024-05-14

Bug Fixes

-  Fixed: Requesting for acceptance of Numbers in the certificate name.

Now Alphanumeric text is accepted in both Certificate Type and Name.

## Release notes: January 05, 2024 Release Notes

Article: https://learningchainai.zendesk.com/hc/en-gb/articles/32445557151129-January-05-2024-Release-Notes

Updated: 2024-05-14

Bug Fixes

-  Fixed: Issue in Bulk generation the "Next button" is active without any files uploaded.

Now the “Next” button is not active.

-  Fixed: Institution User/Admin - Print button on the confirmation page.
Now the “LearningChain LOGO” and “Institution LOGO” is visible properly.

## Release notes: November 17th, 2023 Release Notes

Article: https://learningchainai.zendesk.com/hc/en-gb/articles/28870428354201-November-17th-2023-Release-Notes

Updated: 2024-05-14

New features

-  New: Requesting to display the College Name, Logo, and LearningChain logo on the verification summary print page.

Now, on the verification print page, the College Name with Logo and LearningChain Logo wil be displayed

-  New: Displaying the verifier's name in the admin login

Now, the verifier name is displayed on the Receipts page of the admin login.

## Release notes: October 17th, 2023 Release Notes

Article: https://learningchainai.zendesk.com/hc/en-gb/articles/28870527968793-October-17th-2023-Release-Notes

Updated: 2024-02-17

Bug Fixes

- Fixed: Issue with a validated student in Certificate Verification.

Extra spaces are automatically eliminated when copying and pasting the data from the email into the student's details on the verification page.

- Fixed: Institution User - Blockchain Credentials Verification in verify certificate.

If we upload the certificate or any other document in the place of the QR code we will get and display the message as “No-QR-Code found”.

- Fixed: Institution User/Admin - Action dropdown in Generated and Verified Certificates.

The “GO” button beside the action menu will be enabled only if we select any of the options, or else the “GO” button will be disabled by default.

- Fixed: Issue with Bulk Generation of Certificates and Validation.

When we try to upload exciting files in Blockchain with different details in Bulk Generation we can see the Failure Reason as “The uploaded file already Exist”.

We can view this from the “Reports Tab” -> “Bulk Generation Details.”

- Fixed: Request for Search Option in Printer Case.

Now we have the option to search for student details as required in the printer case generation of the student's certificates.

- Fixed: Verification Process – Steps Reordering Request

The steps for verification were rearranged at the client's request, with the “Upload Document” to the 2nd step and the “Make Payment” moved to the 3rd step.

- Fixed: Naming Convention Modification in Verification Page.

Now, the naming convention is Modified from the “Download Key File” to “Download QR Code” on the verification page.

## Release notes: September 21st, 2023 Release Notes

Article: https://learningchainai.zendesk.com/hc/en-gb/articles/32154674783641-September-21st-2023-Release-Notes

Updated: 2024-05-07

Bug Fixes

• Fixed: Institute User-Select Checkboxes issue in Bulk Generation.

Now, while uploading data in bulk, if there is any Duplicate data and after selecting only one student data, the toggle button does not turn on automatically.

• Fixed: Issue with Data filtering in the Reports Tab.

Now Data filtering is working as expected based on the Filtering of Certificate-Type and Certificate Name.

• Fixed: Issue with visibility of generated date in the dashboard page and report tab.

Now Generated date is visible in the both Dashboard and Reports tab.

-  Fixed: Request for QR code naming Convention adjustment from IARE college.

We got a new update i.e.; we added an underscore “_” between two names of the QR code.

-  Fixed: Request to modify source file and key file Naming Convention.

Modified “Download Source File” to “Download Original Certificate” and “Download Key File” to “Download QR Code” in all the respected places.

-  Fixed: Request from CMR College to include a text field on the verification Page.

Disclaimer Text option has been added and admin has control to enable it. This text will be displayed when anyone clicks on the verification URL

-  Fixed: Institution User - Guidelines option in Single/Bulk Generations.

Before Uploading Certificates user needs to read and agree to the Guidelines check box.

-  Fixed: Request from IARE College for modifications in Visual Verification Page after scanning the QR code.

The modifications are done as follows:
1. "totalCredits" to "Total Credits"
2. "SNO" to "S.No"

New Features

-  New: Need to Provide edit operation to the Email ID field in the Grids.

Now User and admin have access to modify or update the student's email ID

## Release notes: August 31st, 2023 Release Notes

Article: https://learningchainai.zendesk.com/hc/en-gb/articles/32235229133209-August-31st-2023-Release-Notes

Updated: 2024-05-09

Bug Fixes

- Fixed: The issue with the toggle button in stage and production environment.

- Fixed: Institution User - Instant mail after a single generation.

- Fixed: Institution User & Admin - View Verify Certificate page.

- Fixed: Institution User & Admin - Single/Bulk Generation uploaded Data is not displayed in Dashboard and Generated Certificates.

- Fixed: Stage - Prod - Institution User/Institution Admin - Recently Generated Certificate is not showing at the top of the list.

- Fixed: Stage - Prod - Institution User - After Generating the certificate, when we scan the QR code with Scanner, the Email and Mobile number fields are missing.

- Fixed: Stage - Prod - Institution User/Admin - Spelling mistake in Blockchain Credentials Verification page.

## Release notes: August 29th, 2023 Release Notes

Article: https://learningchainai.zendesk.com/hc/en-gb/articles/32235327031321-August-29th-2023-Release-Notes

Updated: 2024-05-09

Bug Fixes

- Fixed: Request to Modify Pop-Up Message for QR Code Mail Sending from Generated Certificate page

- Fixed: The issue with Email Functionality in the dashboard in the production environment

- Fixed: Request from IARE college to display SGPA for Failed Students as blank

- Fixed: Request from IARE college for QR Code Student Information Retrieval

- Fixed: Issue with Verification Process of CMR college in Production Environment

## Release notes: August 11th, 2023 Release Notes

Article: https://learningchainai.zendesk.com/hc/en-gb/articles/32235323244697-August-11th-2023-Release-Notes

Updated: 2024-05-09

Bug Fixes

- Fixed: Spelling Mistake ("Dupicate" instead of "Duplicate").

- Fixed: Page Size dropdown Change in generations grid in reports tab: Added 500,1000-page size in the dropdown as per the new requirement

- Fixed: Institution User and Admin - Rarely, Files are missing in Bulk with Approval.

- Fixed: The issue of Bulk Generation in the production environment

- Fixed: File Upload Problem in Bulk Upload on production environment - Files Not Reflecting on Dashboard

## Release notes: August 3rd, 2023 Release Notes

Article: https://learningchainai.zendesk.com/hc/en-gb/articles/32235431462809-August-3rd-2023-Release-Notes

Updated: 2024-05-09

New Features:

-  New: Page Size dropdown Change in generations grid in reports tab: We have added a dropdown to show the number of generations to be displayed based on the selection of the dropdown.

-  New: Adding Search based on student Email in Generations Grid in Reports: We have added email search and also showing the column in the grid.

## Release notes: July 27th, 2023 Release Notes

Article: https://learningchainai.zendesk.com/hc/en-gb/articles/32235477704601-July-27th-2023-Release-Notes

Updated: 2024-05-09

Bug Fixes

-  Fixed: CR - New Functionality Request for CMR College

Modifications:

Step 1: Please add this point "Note: You are informed to provide the following URL to the verifier for your educational certificate verification purpose" in the above URL for Online student education verification request.

Step 2:  "Authority" should be used in place of "The Principal"

## Release notes: July 7th, 2023 Release Notes

Article: https://learningchainai.zendesk.com/hc/en-gb/articles/32235553643801-July-7th-2023-Release-Notes

Updated: 2024-05-09

New Features:

- New: CR-New document Types: We have implemented a new change ie., we are allowing new document types such as jpeg/jpg, docx, and doc along with pdf document type.

- New: Need to add settings to include with logo/without logo option in the institution and based on that need to generate the logo in QR code

- New: Need to add logo in QR code in single, bulk, Printer case generations Based on with logo/without logo option in institution settings need to generate the logo in QR code.

Bug Fixes

- Fixed: Select toggle button issue in bulk generation screen.

- Fixed: Institution User - Error message for uploading .png file in the place of pdf/doc file upload in the Verify Certificates Page.

- Fixed: Institution User - Issue while downloading the original file in the Confirmation Page in the Verification screen.

- Fixed: Institution User - Error message for uploading CSV before the PDF upload in the Bulk Certification Generation Page.

- Fixed: Institution User - Issues for Recent printer Generations are not reflected in the dashboard.

- Fixed: Institution User - Generation Mail is not receiving while single generation without approval.

- Fixed: Issue on Downloading QR code from E-Mail

## Release notes: April 5th 2023 Release Notes

Article: https://learningchainai.zendesk.com/hc/en-gb/articles/32236391658137-April-5th-2023-Release-Notes

Updated: 2024-05-09

Bug Fixes

- Fixed: Institution - Login session not maintained after not logging out and closing the browser.

- Fixed: Institution - 2 logins (i.e., Institution User & Institution Admin) are opening simultaneously in a Single Browser.

- Fixed: Institution User - After Successful payment, it is not redirecting to upload document in Verify Certificate.

- Fixed: Select All checkbox is not working from 2nd page in Generated and Verified certificates in the reports tab.

- Fixed: Institution Admin - Certificate Type and Certification name dropdowns.

- Fixed: All Roles - Popup is submitting on clicking any place in screens without clicking Ok button.

- Fixed: Select All checkbox is not working in Printer Generation.

- Fixed: Organisation Management/Sales - Rate Contract will be created without entering the Amount Agreed and Amount Paid.

- Fixed: Institution User - View Saved Student Details Data page in Printer Generations.

- Fixed: Institution User & Admin - Select all checkboxes that are not working in View Bulk Generations Uploaded Data page.

- Fixed: Institution User & Admin - Duplicate data found in Approvals.

- Fixed: Institution User - Generate Your Certificate page on the Certificates Tab.

- Fixed: Institution User - Bulk and Printer Generation.

- Fixed: Institution User - CSV File in empty field titles on Bulk upload/Printer Generation.

- Fixed: All Roles - View Payment List Page.

- Fixed: Institution User - Text change View Uploaded Printer Generation Data instead of View Saved Student Details Data in Printer Generations.

- Fixed: Institution Admin - Registration Page.

- Fixed: Organisation Management/Sales - Rate Contract form.

- Fixed: Institution User - After Successful payment, it is not redirecting to upload document in Verify Certificate.

- Fixed: Institution - Logo is broken after creating new institution.

- Fixed: Organisation Management/ Sales - Firefox Browser - Future date is accepted in 'from date' and current date is accepted in 'To Date' in Rate contract/Credits form.

- Fixed: Institution User & Admin - Recently Generated Certificate is not showing at top of the list.

Production issues

- Fixed: Institution Admin - Update Institution is not working.

- Fixed: Institution Admin - Update profile is not working.

- Fixed: Institution User - Latest Bulk generated Certificates not appeared at the Bottom of the page in Bulk Generated Certificates.

New Features:

- New: In Visual verification added animation and also removed option for download Key & Source files

- New: For Printer generations visual verifications added option for download Key & Source files

- New: In Certification Verification added animation

- New: Added Visual Verification Count i.e., whenever qr code is scanned

- New: Split percentage to share amount to institution

- New: QRC Vulnerabilities Fixes

- New: ReCAPTCHA

## Release notes: April 14th, 2023 Release Notes

Article: https://learningchainai.zendesk.com/hc/en-gb/articles/32236137394073-April-14th-2023-Release-Notes

Updated: 2024-05-09

Bug Fixes

- Fixed: Remove the payments option/menu item in Inst User and it should be under Inst Admin. And also modify the menu name item name to Receipts.

- Fixed: Modify the Payments menu item to Receipts and also modify the View Payments list name to Receipts

- Fixed: Show the visual verification count in the respective grid

- Fixed: Remove complete credits block

- Fixed: Show the payments by institution wise

- Fixed: Institution User & Admin - Student Name link in Recent Certificate Generations on the Dashboard / Generated Certificates on the Reports.

- Fixed: In Printer Generation when we try to upload 215 Student's data, View Bulk Generations Uploaded Data page continuously loads. It is not moving to the next page.

- Fixed: When we upload the uppercase letters in the filename, but system only accepts file names in lowercase letters.

- Fixed: Issue on Certificates uploading process in the production environment

- Fixed: Issue on Bulk Generation in the production environment

- Fixed: The last name field is empty on the Update profile page.

- Fixed: Uploading Filename issue in the production environment

- Fixed: Institution User - Latest Bulk generated Certificates did not appear in the Bottom of the page in Bulk Generated Certificates on the Reports Tab.

- Fixed: Institution User & Admin - View Verified Certificates page on the Reports.

New Features:

- New: Added visual verification count

- New: In the grid show common fields for both postpaid and prepaid, then show a link to show the details of the corresponding payment model.

- New: Change the menu item name to Internal Printing also need to set this as a privilege in institution settings and accordingly show the menu item.

## Release notes: May 22nd, 2023 Release Notes

Article: https://learningchainai.zendesk.com/hc/en-gb/articles/32236022987417-May-22nd-2023-Release-Notes

Updated: 2024-05-09

Bug Fixes

- Fixed: Without uploading files and CSV, it will redirect to the next page

- Fixed: current year and current month validation in the verification process

- Fixed: scroll bars in rate contract and credits

- Fixed: Issue on Institution Share Amount Limit in Production Environment

- Fixed: Issue on Uploading the QR code file in the Verification process

- Fixed: Issue when attempting to upload a CSV file followed by a PDF file in the Bulk upload process.

- Fixed: Without checking the checkbox, it will not show any error like please check at least one checkbox

- Fixed: If we put a hyphen (-) and space in the file name and click on Download Key File it shows as a "Blob Not Found" Error.

- Fixed: Facing Issue while downloading QR code file.

- Fixed: single approval files displayed in bulk approval

- Fixed: Issue when attempting to upload a CSV file followed by a PDF file and clicking the next button it shows file not found

- Fixed: The issue with Chrome Learning Chain application

- Fixed: Issue on Bulk Approval process for CMR College

- Fixed: File size changes in single and bulk generation

New Features:

- New: Location tracking for each QR code scan

- New: download files option in generation details based on institution settings

- New: Polygon network in the institution creation page

- New: Rate contract summary and uploaded file in rate-contract creation

- New: Login Impersonation -- login to another user

- New: Added logging in NodeJS

- New: CR-Get the transactions with UID for all the certificates

- New: Added no of files uploading in loader image in bulk generations

## Release notes: June 8th, 2023 Release Notes

Article: https://learningchainai.zendesk.com/hc/en-gb/articles/32236001031705-June-8th-2023-Release-Notes

Updated: 2024-05-09

New Features:

-  New: We have deployed a "Marks Memo" for IARE (Institute of Aeronautical Engineering).

-  New: New Fields added are "S No, Total Credits, SGPA, CGPA" which will be shown for the institution "IARE" only when the certificate type is specifically "BTech" in a single generation. For other institutions, this will not be impacted.

-  New: we have added a new feature for the QR Code File Name Convention in institution settings, based on the selection we will update the QR code file name (please find the below screenshots for your reference).

## Release notes: June 16th, 2023 Release Notes

Article: https://learningchainai.zendesk.com/hc/en-gb/articles/32235909637913-June-16th-2023-Release-Notes

Updated: 2024-05-09

Bug Fixes

-  Fixed: Institution User & Admin - Generated and verified Certificates count is not displayed properly.

-  Fixed: Internal Printing - Unable to download QR Codes.

## Release notes: June 23rd, 2023 Release Notes

Article: https://learningchainai.zendesk.com/hc/en-gb/articles/32235600925977-June-23rd-2023-Release-Notes

Updated: 2024-05-09

Bug Fixes

-  Fixed: CR- Need to add default text below QR Code
