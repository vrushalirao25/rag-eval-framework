# AI Model Risk Management Policy

## Model Inventory
Every AI and machine learning model, including third-party generative AI models accessed through an API, must be registered in the model inventory before it is used in production.
Each inventory entry must record the model owner, intended use, risk tier and validation status.

## Risk Tiering
Models are classified into three risk tiers.
Tier 1 models are customer-facing or influence credit decisions, and require independent validation every year.
Tier 2 models support internal decisions, and require independent validation every 2 years.
Tier 3 models are low-impact productivity tools, and require an annual self-assessment by the model owner.

## Generative AI Controls
Customer-facing content produced by a Tier 1 generative AI system must be reviewed by a human before it is sent to a customer.
Customer personal data must not be included in prompts to externally hosted models unless the Data Protection Officer has approved the use case.
Generative AI systems must be evaluated against a documented golden test set before each production release.

## Model Monitoring
Production models must be monitored for performance drift every month.
Any breach of an agreed performance threshold must be escalated to the Model Risk Committee within 5 business days.
