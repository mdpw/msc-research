# Glossary of Abbreviations and Acronyms

This glossary lists all abbreviations and acronyms used across the chapters of this report. Terms are listed alphabetically.

---

| Abbreviation | Full Form | Context |
|---|---|---|
| AAC | Advanced Audio Coding | Audio codec (referenced as alternative audio format) |
| AI | Artificial Intelligence | General field encompassing ML, NLU, and STT technologies used in this project |
| API | Application Programming Interface | Communication interface between the Android app, backend server, and dashboard |
| APK | Android Package Kit | Distributable file format for installing the guest app on Android tablets |
| ASGI | Asynchronous Server Gateway Interface | Server specification that allows FastAPI and Uvicorn to handle async WebSocket connections |
| AWS | Amazon Web Services | Amazon's cloud infrastructure platform; cloud-based alternatives such as Alexa for Hospitality run on AWS |
| BERT | Bidirectional Encoder Representations from Transformers | Pre-trained transformer model used as the base for intent classification fine-tuning |
| BCS | British Computer Society | Professional body whose code of conduct informs the ethical standards of this project |
| BOM | Bill of Materials | Version specification file used by Jetpack Compose to manage compatible library versions |
| CER | Character Error Rate | Metric measuring the percentage of incorrectly transcribed characters in Vosk output |
| CLI | Command-Line Interface | Text-based interface used to start the backend server and run training scripts |
| CPU | Central Processing Unit | Processing hardware on which the on-device models run (no GPU used) |
| CRUD | Create, Read, Update, Delete | Standard database operations; the FastAPI backend CRUD layer uses raw SQL without an ORM |
| CSS | Cascading Style Sheets | Styling language used for the staff dashboard UI |
| CSV | Comma-Separated Values | File format for storing training datasets (e.g., `new_hotel_dataset.csv`) |
| DevOps | Development and Operations | Integrated approach combining development and deployment practices |
| DIET | Dual Intent and Entity Transformer | Intent classification architecture used in the Rasa framework (reviewed in literature) |
| DistilBERT | Distilled BERT | Lightweight BERT variant created via knowledge distillation (reviewed in literature) |
| DSR | Design Science Research | Research methodology framework adopted for this project |
| ERM | Entity-Relationship Model | Diagram showing the database schema and table relationships |
| FR | Functional Requirement | Prefix used to label system functional requirements (FR-01 through FR-10) |
| GDPR | General Data Protection Regulation | EU data protection regulation relevant to processing guest voice data |
| GLUE | General Language Understanding Evaluation | Benchmark suite used to evaluate transformer model performance (referenced in literature) |
| GPIO | General Purpose Input/Output | Hardware interface on single-board computers (referenced in IoT deployment context) |
| GPU | Graphics Processing Unit | Hardware accelerator used in training but not in on-device inference |
| gTTS | Google Text-to-Speech | Python library used in `step2_generate_vosk_noise.py` to synthesise audio from text utterances for Vosk pairing |
| HCI | Human-Computer Interaction | Design discipline informing the voice-first UI/UX decisions |
| HTML | HyperText Markup Language | Markup language used for the staff dashboard (single self-contained HTML file) |
| HTTP | Hypertext Transfer Protocol | Protocol used for REST API communication between the Android app and server |
| HTTPS | Hypertext Transfer Protocol Secure | Encrypted version of HTTP (recommended for production deployment) |
| IDE | Integrated Development Environment | Development tool; Android Studio used for Android development |
| INT8 | 8-bit Integer | Numeric format used in model quantisation; dynamic range quantisation converts model weights to INT8 |
| IoT | Internet of Things | Domain of edge-deployed smart devices relevant to on-device AI deployment |
| IP | Internet Protocol | Network-layer protocol; the server is accessed by its local IP address |
| JSON | JavaScript Object Notation | Data format used for API responses, label maps, and vocabulary files |
| JWT | JSON Web Token | Token-based authentication format (noted as future enhancement) |
| KPI | Key Performance Indicator | Measurable metric used to evaluate system performance |
| LAN | Local Area Network | Hotel's internal Wi-Fi network on which the entire system runs |
| MB | Megabyte | Unit of data storage (Vosk model: ~36 MB; NLU model: 26 MB) |
| MDM | Mobile Device Management | Platform for centralised tablet provisioning (noted as future enhancement) |
| MFCC | Mel-Frequency Cepstral Coefficients | Acoustic features used by the Vosk speech recognition model |
| ML | Machine Learning | Subfield of AI; used for STT and intent classification models |
| MobileBERT | Mobile BERT | Compact BERT variant optimised for mobile deployment; used as the NLU backbone |
| MoSCoW | Must have, Should have, Could have, Won't have | Prioritisation framework used to classify requirements |
| MQTT | Message Queuing Telemetry Transport | Lightweight IoT messaging protocol (reviewed as alternative to WebSocket) |
| MVP | Minimum Viable Product | Scope descriptor for the research prototype |
| NFC | Near Field Communication | Short-range wireless technology (noted as future option for room pairing) |
| NFR | Non-Functional Requirement | Prefix used to label performance and quality requirements (NFR-01 through NFR-07) |
| NLU | Natural Language Understanding | Component responsible for classifying guest intent from transcribed speech |
| ONNX | Open Neural Network Exchange | Cross-framework model format; considered for Whisper Android deployment but not adopted |
| ORM | Object-Relational Mapping | Database abstraction layer deliberately excluded from the backend (raw SQL used instead) |
| P95 | 95th Percentile | Latency metric indicating the response time below which 95% of requests fall (P95 = 2,827 ms) |
| PCM | Pulse Code Modulation | Audio encoding format; the app captures audio at 16 kHz, 16-bit PCM mono |
| PDPA | Personal Data Protection Act | Sri Lanka's data protection legislation (Personal Data Protection Act No. 9 of 2022) |
| PMS | Property Management System | Hotel operations software (e.g., Opera, Mews); noted as a future integration target |
| PNG | Portable Network Graphics | Image format used for evaluation output (confusion matrix plots) |
| POS | Point of Sale | Hotel retail system noted as a future integration point |
| QA | Quality Assurance | Process of verifying that the system meets defined requirements |
| QR | Quick Response | QR code format noted as a future option for automated device provisioning |
| RAG | Retrieval-Augmented Generation | AI technique for grounding LLM responses in a knowledge base (future enhancement) |
| RAM | Random Access Memory | Device memory; budget tablet constraints influenced the choice of the 36 MB Vosk model |
| REST | Representational State Transfer | Architectural style for the HTTP API between the Android app and the backend |
| RMS | Root Mean Square | Energy measurement used in the VAD algorithm to detect speech (threshold: 0.02) |
| SDK | Software Development Kit | Native libraries included in the Android app (Vosk SDK, TFLite SDK) |
| SLTDA | Sri Lanka Tourism Development Authority | Government body that regulates and promotes Sri Lankan tourism |
| SQL | Structured Query Language | Language used directly for all database queries (no ORM layer) |
| SSL | Secure Sockets Layer | Predecessor to TLS; used informally to refer to encrypted connections |
| STT | Speech-to-Text | The process of converting recorded audio to a text transcript using the Vosk model |
| TCO | Total Cost of Ownership | Three-year cost projection used in the cost–benefit analysis |
| TFLite | TensorFlow Lite | Lightweight ML runtime used to run the MobileBERT model on Android |
| TLS | Transport Layer Security | Encryption protocol used by HTTPS (recommended for production) |
| TTS | Text-to-Speech | Converts text responses to spoken audio on the guest device (Android native engine) |
| UAT | User Acceptance Testing | User-facing evaluation confirming the system meets guest-side requirements |
| UI | User Interface | Visual and interactive elements of both the guest app and the staff dashboard |
| UML | Unified Modeling Language | Diagramming standard used for use case and sequence diagrams in Chapters 5 and 6 |
| USD | United States Dollar | Currency used in cost estimates and market data cited in the literature review |
| UX | User Experience | Quality of the overall guest interaction with the voice assistant |
| VAD | Voice Activity Detection | Energy-based algorithm that determines when the guest has started and stopped speaking |
| VLAN | Virtual Local Area Network | Network segmentation technique recommended for production security hardening |
| WER | Word Error Rate | Metric measuring the percentage of incorrectly transcribed words in Vosk output |
| WTTC | World Travel and Tourism Council | International tourism body; cited for global hospitality industry statistics |

---

*Total entries: 70*
