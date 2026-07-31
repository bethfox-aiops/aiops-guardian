# NIST AI RMF 1.0 — Reference

This document is Stage 1 of mapping Guardian's Behavioral Attestation work to
the NIST AI Risk Management Framework (`ROADMAP.md` Phase 7 item 3). It is a
**reference**, not a mapping — it captures the framework's actual structure,
verbatim, so later stages have an accurate primary source to work from
instead of re-reading the PDF each time or working from paraphrase.

**Source:** NIST AI 100-1, "Artificial Intelligence Risk Management Framework
(AI RMF 1.0)," January 2023. Full document:
https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf — free, public,
voluntary (not a certification). Companion practical guidance: the AI RMF
Playbook, https://airc.nist.gov/airmf-resources/playbook/.

## The four functions, at a glance

- **GOVERN** — cross-cutting; cultivates a culture of risk management,
  infused throughout the other three functions rather than a discrete step.
- **MAP** — establishes context: what is this AI system, who does it affect,
  what could go wrong.
- **MEASURE** — quantitative/qualitative testing, tracking, and monitoring of
  identified risks.
- **MANAGE** — prioritizes and acts on risks identified by Map and Measure;
  response, recovery, communication.

NIST's own guidance: most users start with Govern, then Map, then iterate
into Measure/Manage — not a strict linear checklist.

## Table 1 — GOVERN

| Category | Subcategories |
|---|---|
| **GOVERN 1**: Policies, processes, procedures, and practices across the organization related to mapping, measuring, and managing AI risks are in place, transparent, and implemented effectively. | **1.1** Legal and regulatory requirements involving AI are understood, managed, and documented. **1.2** The characteristics of trustworthy AI are integrated into organizational policies, processes, procedures, and practices. **1.3** Processes, procedures, and practices are in place to determine the needed level of risk management activities based on the organization's risk tolerance. **1.4** The risk management process and its outcomes are established through transparent policies, procedures, and other controls based on organizational risk priorities. **1.5** Ongoing monitoring and periodic review of the risk management process and its outcomes are planned and organizational roles and responsibilities clearly defined, including determining the frequency of periodic review. **1.6** Mechanisms are in place to inventory AI systems and are resourced according to organizational risk priorities. **1.7** Processes and procedures are in place for decommissioning and phasing out AI systems safely and in a manner that does not increase risks or decrease the organization's trustworthiness. |
| **GOVERN 2**: Accountability structures are in place so that the appropriate teams and individuals are empowered, responsible, and trained for mapping, measuring, and managing AI risks. | **2.1** Roles and responsibilities and lines of communication related to mapping, measuring, and managing AI risks are documented and are clear to individuals and teams throughout the organization. **2.2** The organization's personnel and partners receive AI risk management training to enable them to perform their duties and responsibilities consistent with related policies, procedures, and agreements. **2.3** Executive leadership of the organization takes responsibility for decisions about risks associated with AI system development and deployment. |
| **GOVERN 3**: Workforce diversity, equity, inclusion, and accessibility processes are prioritized in the mapping, measuring, and managing of AI risks throughout the lifecycle. | **3.1** Decision-making related to mapping, measuring, and managing AI risks throughout the lifecycle is informed by a diverse team. **3.2** Policies and procedures are in place to define and differentiate roles and responsibilities for human-AI configurations and oversight of AI systems. |
| **GOVERN 4**: Organizational teams are committed to a culture that considers and communicates AI risk. | **4.1** Organizational policies and practices are in place to foster a critical thinking and safety-first mindset in the design, development, deployment, and uses of AI systems to minimize potential negative impacts. **4.2** Organizational teams document the risks and potential impacts of the AI technology they design, develop, deploy, evaluate, and use, and they communicate about the impacts more broadly. **4.3** Organizational practices are in place to enable AI testing, identification of incidents, and information sharing. |
| **GOVERN 5**: Processes are in place for robust engagement with relevant AI actors. | **5.1** Organizational policies and practices are in place to collect, consider, prioritize, and integrate feedback from those external to the team that developed or deployed the AI system regarding the potential individual and societal impacts related to AI risks. **5.2** Mechanisms are established to enable the team that developed or deployed AI systems to regularly incorporate adjudicated feedback from relevant AI actors into system design and implementation. |
| **GOVERN 6**: Policies and procedures are in place to address AI risks and benefits arising from third-party software and data and other supply chain issues. | **6.1** Policies and procedures are in place that address AI risks associated with third-party entities, including risks of infringement of a third-party's intellectual property or other rights. **6.2** Contingency processes are in place to handle failures or incidents in third-party data or AI systems deemed to be high-risk. |

## Table 2 — MAP

| Category | Subcategories |
|---|---|
| **MAP 1**: Context is established and understood. | **1.1** Intended purposes, potentially beneficial uses, context-specific laws, norms and expectations, and prospective settings in which the AI system will be deployed are understood and documented. **1.2** Interdisciplinary AI actors, competencies, skills, and capacities for establishing context reflect demographic diversity and broad domain and user experience expertise, and their participation is documented. **1.3** The organization's mission and relevant goals for AI technology are understood and documented. **1.4** The business value or context of business use has been clearly defined or — in the case of assessing existing AI systems — re-evaluated. **1.5** Organizational risk tolerances are determined and documented. **1.6** System requirements are elicited from and understood by relevant AI actors. Design decisions take socio-technical implications into account. |
| **MAP 2**: Categorization of the AI system is performed. | **2.1** The specific tasks and methods used to implement the tasks that the AI system will support are defined. **2.2** Information about the AI system's knowledge limits and how system output may be utilized and overseen by humans is documented. **2.3** Scientific integrity and TEVV (test, evaluation, verification, validation) considerations are identified and documented. |
| **MAP 3**: AI capabilities, targeted usage, goals, and expected benefits and costs compared with appropriate benchmarks are understood. | **3.1** Potential benefits of intended AI system functionality and performance are examined and documented. **3.2** Potential costs, including non-monetary costs, which result from expected or realized AI errors or system functionality and trustworthiness are examined and documented. **3.3** Targeted application scope is specified and documented. **3.4** Processes for operator and practitioner proficiency with AI system performance and trustworthiness are defined, assessed, and documented. **3.5** Processes for human oversight are defined, assessed, and documented in accordance with organizational policies from the GOVERN function. |
| **MAP 4**: Risks and benefits are mapped for all components of the AI system including third-party software and data. | **4.1** Approaches for mapping AI technology and legal risks of its components — including the use of third-party data or software — are in place, followed, and documented. **4.2** Internal risk controls for components of the AI system, including third-party AI technologies, are identified and documented. |
| **MAP 5**: Impacts to individuals, groups, communities, organizations, and society are characterized. | **5.1** Likelihood and magnitude of each identified impact are identified and documented. **5.2** Practices and personnel for supporting regular engagement with relevant AI actors and integrating feedback about positive, negative, and unanticipated impacts are in place and documented. |

## Table 3 — MEASURE

| Category | Subcategories |
|---|---|
| **MEASURE 1**: Appropriate methods and metrics are identified and applied. | **1.1** Approaches and metrics for measurement of AI risks enumerated during the MAP function are selected for implementation starting with the most significant AI risks. **1.2** Appropriateness of AI metrics and effectiveness of existing controls are regularly assessed and updated. **1.3** Internal experts who did not serve as front-line developers and/or independent assessors are involved in regular assessments and updates. |
| **MEASURE 2**: AI systems are evaluated for trustworthy characteristics. | **2.1** Test sets, metrics, and details about the tools used during TEVV are documented. **2.2** Evaluations involving human subjects meet applicable requirements and are representative of the relevant population. **2.3** AI system performance or assurance criteria are measured qualitatively or quantitatively and demonstrated for conditions similar to deployment setting(s). **2.4** The functionality and behavior of the AI system and its components are monitored when in production. **2.5** The AI system to be deployed is demonstrated to be valid and reliable. **2.6** The AI system is evaluated regularly for safety risks. **2.7** AI system security and resilience are evaluated and documented. **2.8** Risks associated with transparency and accountability are examined and documented. **2.9** The AI model is explained, validated, and documented, and AI system output is interpreted within its context. **2.10** Privacy risk of the AI system is examined and documented. **2.11** Fairness and bias are evaluated and results are documented. **2.12** Environmental impact and sustainability of AI model training and management activities are assessed and documented. **2.13** Effectiveness of the employed TEVV metrics and processes is evaluated and documented. |
| **MEASURE 3**: Mechanisms for tracking identified AI risks over time are in place. | **3.1** Approaches, personnel, and documentation are in place to regularly identify and track existing, unanticipated, and emergent AI risks. **3.2** Risk tracking approaches are considered for settings where AI risks are difficult to assess using currently available measurement techniques. **3.3** Feedback processes for end users and impacted communities to report problems and appeal system outcomes are established and integrated into AI system evaluation metrics. |
| **MEASURE 4**: Feedback about efficacy of measurement is gathered and assessed. | **4.1** Measurement approaches for identifying AI risks are connected to deployment context(s) and informed through consultation with domain experts and other end users. **4.2** Measurement results regarding AI system trustworthiness in deployment context(s) are informed by input from domain experts and relevant AI actors. **4.3** Measurable performance improvements or declines based on consultations with relevant AI actors are identified and documented. |

## Table 4 — MANAGE

| Category | Subcategories |
|---|---|
| **MANAGE 1**: AI risks based on assessments and other analytical output from the MAP and MEASURE functions are prioritized, responded to, and managed. | **1.1** A determination is made as to whether the AI system achieves its intended purposes and stated objectives and whether its development or deployment should proceed. **1.2** Treatment of documented AI risks is prioritized based on impact, likelihood, and available resources or methods. **1.3** Responses to the AI risks deemed high priority are developed, planned, and documented. Risk response options can include mitigating, transferring, avoiding, or accepting. **1.4** Negative residual risks (defined as the sum of all unmitigated risks) to both downstream acquirers of AI systems and end users are documented. |
| **MANAGE 2**: Strategies to maximize AI benefits and minimize negative impacts are planned, prepared, implemented, documented, and informed by input from relevant AI actors. | **2.1** Resources required to manage AI risks are taken into account. **2.2** Mechanisms are in place and applied to sustain the value of deployed AI systems. **2.3** Procedures are followed to respond to and recover from a previously unknown risk when it is identified. **2.4** Mechanisms are in place and applied, and responsibilities are assigned and understood, to supersede, disengage, or deactivate AI systems that demonstrate performance or outcomes inconsistent with intended use. |
| **MANAGE 3**: AI risks and benefits from third-party entities are managed. | **3.1** AI risks and benefits from third-party resources are regularly monitored, and risk controls are applied and documented. **3.2** Pre-trained models which are used for development are monitored as part of AI system regular monitoring and maintenance. |
| **MANAGE 4**: Risk treatments, including response and recovery, and communication plans for the identified and measured AI risks are documented and monitored regularly. | **4.1** Post-deployment AI system monitoring plans are implemented, including mechanisms for capturing and evaluating input from users and other relevant AI actors, appeal and override, decommissioning, incident response, recovery, and change management. **4.2** Measurable activities for continual improvements are integrated into AI system updates. **4.3** Incidents and errors are communicated to relevant AI actors, including affected communities. Processes for tracking, responding to, and recovering from incidents and errors are followed and documented. |

## The seven trustworthiness characteristics (context for MEASURE 2.x)

Named in the framework as what "trustworthy AI" is actually assessed against
(full definitions in the source PDF, section 3, pages 12-19 — summarized
here, not quoted in full):

1. **Valid and Reliable** — the necessary base condition; everything else is
   assessed on top of it.
2. **Safe** — should not endanger human life, health, property, or the
   environment.
3. **Secure and Resilient** — withstands adversarial conditions (data
   poisoning, exfiltration, unauthorized access) and degrades gracefully
   rather than catastrophically.
4. **Accountable and Transparent** — accountability presupposes
   transparency; information about the system is available to those
   affected by it.
5. **Explainable and Interpretable** — explainability = how the system
   works; interpretability = what its output means in context.
6. **Privacy-Enhanced** — safeguards autonomy, identity, and dignity;
   anonymity, confidentiality, control.
7. **Fair, with harmful bias managed** — covers systemic, computational/
   statistical, and human-cognitive bias, not just demographic balance.

## What this document deliberately does NOT do

This is the reference only. It does not yet say anything about Guardian —
no claims of coverage, partial coverage, or gaps. That's the next stage:
a genuine, honest walk through every subcategory above against
`behavioral_policy.py`, the watchdogs, `aiops-approval.service`, and
Guardian's actual documentation, deciding case by case whether there's
real evidence, partial evidence, or a genuine gap — including being
comfortable marking some subcategories "not applicable at this scale"
(e.g., several MAP/GOVERN subcategories assume an organization with
external stakeholders, which a single-user personal system doesn't have
in the same way).

See `ROADMAP.md` Phase 7 item 3 for how this fits the broader plan.
