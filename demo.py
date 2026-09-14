"""
Hiver AI Support Agent - End-to-End System Demonstration (M1 - M7)
Demonstrates the complete pipeline:
- M2: NLP Decontraction & Normalization
- M3: BM25 Knowledge Base Indexing & Retrieval
- M4: Intent Confidence Gating
- M5: Guardrails & PII Masking (Luhn Algorithm)
- M6: Draft Collision & Concurrency Locking
- M7: Business Telemetry & HITL Value-per-Cost Dashboard
"""

from datetime import datetime, timezone, timedelta
import sys

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# M2 NLP
from src.nlp.decontraction import expand_contractions

# M3 Retrieval
from src.retrieval.schema import RetrievalDocument
from src.retrieval.bm25_index import build_index, query_index

# M4 Intent Confidence Gating
from src.models.intent.confidence_gate import apply_confidence_gate

# M5 Guardrails / PII Redaction
from src.guardrails.pii_masking import mask_pii

# M6 Draft Collision Locking
from src.integrations.hiver.collision import (
    register_local_lock,
    get_local_lock_age_minutes,
    deregister_local_lock,
)

# M7 Business Telemetry & HITL Orchestrator
from src.telemetry.schema import ContainmentEvent, HITLReviewEvent
from src.telemetry.telemetry_dashboard import build_telemetry_dashboard


def run_pipeline():
    print("=" * 70)
    print("[*] HIVER AI CUSTOMER SUPPORT AGENT - END-TO-END DEMO (M1-M7)")
    print("=" * 70)

    # 1. Incoming customer query
    raw_query = "I can't access my flight booking, card 4111 1111 1111 1111 was charged!"
    print(f"\n[1] INCOMING CUSTOMER MESSAGE:\n    \"{raw_query}\"")

    # 2. M2: NLP Decontraction
    normalized_query = expand_contractions(raw_query)
    print(f"\n[2] M2: NLP DECONTRACTION:\n    \"{normalized_query}\"")

    # 3. M5: Guardrails & PII Masking (Luhn Check)
    masked_query, detected_pii = mask_pii(normalized_query)
    print(f"\n[3] M5: GUARDRAILS & PII MASKING (Luhn Algorithm):")
    print(f"    Masked Text: \"{masked_query}\"")
    print(f"    Detected PII Types: {detected_pii}")

    # 4. M4: Intent Confidence Gating
    print(f"\n[4] M4: INTENT CONFIDENCE GATING:")
    candidates = [("billing_and_payments", 0.94), ("account_access", 0.32)]
    gated_results = apply_confidence_gate(candidates, threshold=0.65)
    for (intent, conf), passed in zip(candidates, gated_results):
        status = "PASSED (Automate)" if passed else "FAILED (Escalate to HITL)"
        print(f"    Intent: {intent:<22} | Conf: {conf:.2f} | Status: {status}")

    # 5. M3: Knowledge Base Retrieval (BM25)
    print(f"\n[5] M3: BM25 KNOWLEDGE BASE RETRIEVAL:")
    docs = [
        RetrievalDocument(
            doc_id="kb_billing_01",
            query_text="flight booking charge payment dispute refund card billing",
            resolution_text="To dispute or refund a charge for a booking, verify your reference code in the billing portal.",
        ),
        RetrievalDocument(
            doc_id="kb_account_02",
            query_text="reset password account login credentials access",
            resolution_text="Click on 'Forgot Password' on the login screen to receive an SMS reset link.",
        ),
        RetrievalDocument(
            doc_id="kb_general_03",
            query_text="office hours customer support contact phone",
            resolution_text="Our support hours are 24/7 across all live channels.",
        ),
    ]
    index = build_index(docs, k1=1.5, b=0.75)
    doc_map = {d.doc_id: d.resolution_text for d in docs}
    search_results = query_index(index, "flight booking charged card", top_k=2)
    for rank, res in enumerate(search_results, start=1):
        print(f"    Rank #{rank}: Doc ID [{res.doc_id}] (Score: {res.score:.3f})")
        print(f"    Suggested Resolution: \"{doc_map.get(res.doc_id, '')}\"")

    # 6. M6: Draft Collision & Lock Management
    thread_id = "th_demo_98214"
    print(f"\n[6] M6: DRAFT COLLISION LOCKING (Thread: {thread_id}):")
    register_local_lock(thread_id, agent_id="agent_ai_bot")
    lock_age = get_local_lock_age_minutes(thread_id)
    print(f"    Local lock acquired by: agent_ai_bot (Age: {lock_age:.2f} mins)")
    deregister_local_lock(thread_id)
    print(f"    Local lock cleanly released.")

    # 7. M7: Unified Business Telemetry & HITL Value Dashboard
    print(f"\n[7] M7: UNIFIED BUSINESS TELEMETRY & HITL VALUE DASHBOARD:")
    now = datetime.now(timezone.utc)
    containment_events = [
        # Contained + Verified via high CSAT
        ContainmentEvent(
            thread_id="th_001",
            customer_id="cust_1",
            contained=True,
            csat_score=4.8,
            had_repeat_contact_within_48h=False,
            first_response_at=now - timedelta(hours=50),
        ),
        # Contained + Verified via absence of repeat contact
        ContainmentEvent(
            thread_id="th_002",
            customer_id="cust_2",
            contained=True,
            csat_score=None,
            had_repeat_contact_within_48h=False,
            first_response_at=now - timedelta(hours=60),
        ),
        # Contained but Hallucinated/Unresolved (Repeat contact within 48h)
        ContainmentEvent(
            thread_id="th_003",
            customer_id="cust_3",
            contained=True,
            csat_score=2.0,
            had_repeat_contact_within_48h=True,
            first_response_at=now - timedelta(hours=55),
        ),
        # Escalated to Human (Not contained)
        ContainmentEvent(
            thread_id="th_004",
            customer_id="cust_4",
            contained=False,
            csat_score=4.5,
            had_repeat_contact_within_48h=False,
            first_response_at=now - timedelta(hours=10),
        ),
    ]

    hitl_events = [
        HITLReviewEvent(
            thread_id="th_hitl_01",
            escalation_tier="tier_1",
            reviewer_minutes=4.0,
            sla_target_minutes=10,
            actual_response_delay_minutes=5.0,
            error_caught=True,
            error_severity="high",
        ),
        HITLReviewEvent(
            thread_id="th_hitl_02",
            escalation_tier="tier_2",
            reviewer_minutes=6.0,
            sla_target_minutes=15,
            actual_response_delay_minutes=8.0,
            error_caught=True,
            error_severity="critical",
        ),
    ]

    dashboard = build_telemetry_dashboard(
        containment_events=containment_events,
        hitl_events=hitl_events,
        window_label="Live Production Demo",
    )

    print(f"    [+] Window: {dashboard.window_label}")
    print(f"    [+] Headline Metric - Resolution Rate WITHIN Containment: "
          f"{dashboard.resolution_rate_within_containment * 100:.1f}% "
          f"({dashboard.verified_count} verified)")
    print(f"    [+] Secondary Metric - Raw Containment Rate: "
          f"{dashboard.raw_containment_rate * 100:.1f}%")
    print(f"    [+] HITL Operational Cost: ${dashboard.hitl_summary.total_cost:.2f}")
    print(f"    [+] HITL Error Value Saved: ${dashboard.hitl_summary.total_value:.2f}")
    print(f"    [+] HITL Value-per-Cost Ratio: {dashboard.hitl_summary.mean_ratio:.2f}x")

    print("\n" + "=" * 70)
    print("[SUCCESS] PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
