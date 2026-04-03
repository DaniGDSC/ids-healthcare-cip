"""Production deployment services for WUSTL-compatible hospital networks.

Services:
  FlowCollector     — Argus/NetFlow → Kafka producer
  BiometricBridge   — HL7v2/FHIR → Kafka producer
  FeatureFuser      — joins network + biometric → scaled 24-feature vector
  InferenceService  — streaming model inference + risk scoring
"""
