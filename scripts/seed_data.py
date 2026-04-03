"""
scripts/seed_data.py

Loads sample business documents into the FAISS vector store.
Run once after setting up the environment:
    python scripts/seed_data.py
"""

import asyncio
from pathlib import Path
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from core.config import settings

SAMPLE_DOCS = [
    Document(
        page_content="""Q3 2024 Strategy Review — Executive Summary
        
Revenue growth slowed to 8% YoY in Q3 against a target of 14%. APAC was the primary drag,
underperforming by 22% due to delayed enterprise renewals pushed to Q1 2025. AMER held strong
at +12% driven by CloudVault Plus upsells. The board approved a revised Q4 target for APAC
with a focus on mid-market rather than enterprise accounts. Partnership channel expansion in
Southeast Asia is planned for Q1 2025. No headcount changes are planned for Q4.""",
        metadata={"source": "strategy_q3_2024.pdf#page=1", "doc_type": "strategy", "date": "2024-09-30"},
    ),
    Document(
        page_content="""APAC Revenue Decline — Root Cause Analysis (Internal)

The 18% revenue decline in APAC Q4 2024 stems from three factors:
1. Enterprise deal slippage: 3 deals totaling $2.1M moved to Q1 2025 due to procurement freezes
   at two major banking clients in Singapore and Australia.
2. Competitive pressure: Competitor launched a lower-priced tier in October 2024 capturing
   ~15% of our SMB pipeline in Australia.
3. Currency headwinds: AUD depreciation by 6% reduced reported USD revenue.

Mitigation: Account teams have confirmed the slipped deals remain in pipeline with revised
close dates of Feb–March 2025. Pricing committee is reviewing a competitive response.""",
        metadata={"source": "apac_rca_q4_2024.pdf#page=1", "doc_type": "analysis", "date": "2025-01-10"},
    ),
    Document(
        page_content="""Product Roadmap Update — Q4 2024

AnalyticsHub 2.0 launched November 15, 2024 with AI-assisted dashboards. Early adoption is
strong in AMER (210 new seats in 6 weeks) but APAC rollout is delayed due to data residency
requirements in Australia and Singapore — engineering is targeting compliance by Feb 2025.

CloudVault Plus added SOC2 Type II certification in October, unblocking several enterprise
deals in financial services that had been stalled since Q2. The security module contributed
$1.4M in new ARR in Q4.""",
        metadata={"source": "product_roadmap_q4_2024.pdf#page=2", "doc_type": "product", "date": "2024-12-01"},
    ),
    Document(
        page_content="""Sales Team Performance — Q4 2024 Memo

Top performers: AMER Direct team exceeded quota by 18% led by enterprise accounts in
financial services. EMEA partner channel grew 24% QoQ — partner programme expansion is
working. APAC direct team missed quota by 31%; three AEs are on performance improvement
plans. A new APAC sales director started January 6, 2025 — focus will be on rebuilding the
mid-market pipeline which has been under-resourced.

Channel mix: Online self-serve grew to 28% of new ARR (from 19% in Q3), indicating our
product-led growth investment is paying off.""",
        metadata={"source": "sales_memo_q4_2024.pdf#page=1", "doc_type": "memo", "date": "2025-01-05"},
    ),
]


async def main():
    print("Seeding vector store with sample documents...")
    embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)

    index_path = Path(settings.vector_store_path)
    index_path.mkdir(parents=True, exist_ok=True)

    faiss = FAISS.from_documents(SAMPLE_DOCS, embeddings)
    faiss.save_local(str(index_path))

    print(f"✅ Indexed {len(SAMPLE_DOCS)} documents → {index_path}")
    print("\nDocuments indexed:")
    for doc in SAMPLE_DOCS:
        print(f"  • {doc.metadata['source']}")


if __name__ == "__main__":
    asyncio.run(main())
