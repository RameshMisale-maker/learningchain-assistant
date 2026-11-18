# LearningChain Security: Blockchain You Can Trust

Source: https://learningchain.in/security/
Last updated: 2024-11-18

At LearningChain, we prioritize the security and privacy of your information. Our robust Security Policy underscores our unwavering dedication to protecting all proprietary information and entrusted assets. This commitment fosters an environment of operational efficiency, unwavering safety, and robust security for both LearningChain and our valued customers.

### Governance and Compliance

Ensuring security and compliance is the foundation of LearningChain. We set stringent policies and controls in collaboration with our dedicated Security and Privacy teams. These measures are constantly monitored for adherence, and we willingly subject our practices to scrutiny by third-party auditors to validate our security and compliance posture. Our security policies stem from core principles:

- **Principle of Least Privilege:** Access is granted exclusively to individuals with a legitimate 
business need, based on the minimum level necessary.
- **Defense-in-Depth:** Security controls are strategically layered, bolstering protection through 
multiple lines of defense.
- **Uniform Application:** Consistent application of security controls across all sectors of our 
operations.
- **Continuous Enhancement:** Controls are iteratively refined, enhancing effectiveness, 
auditability, and minimizing friction.

### Data Protection

- Data at Rest: Every datastore containing customer data, including Azure Storage, undergoes encryption at rest. Highly sensitive collections and tables employ row-level encryption, ensuring that information is secured even before it reaches the database. This measure guarantees that neither physical access nor logical access to the database is sufficient to access the most sensitive data.
- Data in Transit: LearningChain employs TLS 1.2 or higher to safeguard data transmitted over potentially insecure networks. Our commitment extends further with the application of HSTS (HTTP Strict Transport Security) for enhanced data security during transit. Additionally, Azure-managed TLS keys and certificates are deployed via Application Load Balancers, solidifying our data’s protection.

### Secret Management

Encryption keys are stored in Azure Key Vault (AKV), which employs Hardware Security Modules (HSMs) to shield key material from direct access-ensuring privacy even from Azure and LearningChain employees. These HSM-stored keys facilitate encryption and decryption through Azure’s AKV APIs. Application secrets are equally safeguarded, being encrypted and stored within Azure Key Management Service, with access stringently controlled.

### Product Security

- **Penetration Testing:** LearningChain collaborates annually with esteemed external penetration testing consulting firms. These assessments span all segments of our product and cloud infrastructure, offering testers full access to source code for comprehensive evaluation. Customers can request summary penetration test reports to verify our robust security measures.
- **Vulnerability Scanning:** Our Secure Development Lifecycle (SDLC) incorporates critical vulnerability scanning points:
  - Static analysis (SAST): During code pull requests and continuously.
  - Software composition analysis (SCA): Identifying known software vulnerabilities.
  - Malicious dependency scanning: To thwart malware infiltration.
  - Dynamic analysis (DAST): Of live applications.
  - Network vulnerability scanning: On a scheduled basis.
  - Continuous external attack surface management (EASM): To discover and address new external-facing assets.

### Enterprise Security

- **Endpoint Protection:**Central management of corporate devices includes mobile device 
management software and anti-malware protection. Continuous 24/7/365 monitoring of 
endpoint security alerts is in place, and strict configurations are enforced through MDM 
software, covering disk encryption, screen lock settings, and software updates.
- **Vendor Security:** LearningChain adopts a risk-based approach for vendor security 
assessment, considering factors such as data access, integration with production 
environments, and potential impact on the LearningChain brand. This evaluation determines 
both inherent and residual risk ratings, guiding vendor approval decisions.
- **Secure Remote Access:** LearningChain fortifies remote access to internal resources using 
Azure VPN, while internet browsing is protected by malware-blocking DNS servers for 
employee and endpoint safety.

### Security Education

- **Comprehensive Training:** LearningChain invests in comprehensive security training for all employees, both during onboarding and annually. Tailored educational modules within our platform equip employees with the latest security knowledge.
- **Mandatory Onboarding:** New employees undergo mandatory live onboarding sessions, emphasizing key security and secure coding principles. Regular threat briefings ensure that employees are updated on critical security updates and precautions.

### Identity and Access Management

LearningChain leverages Azure Active Directory to secure identity and access management. Stringent measures include the implementation of phishing-resistant authentication factors, with WebAuthn as a preferred choice whenever possible. Role-based application access is granted to employees and automatically revoked upon termination. Any additional access requires adherence to specific application policies.

### Regulatory Compliance

LearningChain’s dedication to security encompasses a continuous evaluation of regulatory and emerging frameworks, enabling us to adapt and evolve our program accordingly. Our commitment ensures that your data remains protected in line with evolving standards.
