# agent.py — Legal Document Assistant shared agent module
# Import and run: from agent import app, embedder, collection

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List
from datetime import date, timedelta
import chromadb
from sentence_transformers import SentenceTransformer

# ── Constants ──────────────────────────────────────────────
FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES = 2

# ── Models ─────────────────────────────────────────────────
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── Knowledge Base ─────────────────────────────────────────
DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Contract Formation and Elements",
        "text": """A valid contract requires four essential elements: offer, acceptance, consideration, and mutual assent (also called meeting of the minds). Without all four, a court may find the agreement unenforceable. An offer is a clear proposal by one party (the offeror) to another (the offeree) that specifies the essential terms and demonstrates a willingness to be bound. Advertisements are generally not offers but invitations for customers to make offers. Acceptance must be unconditional and mirror the terms of the offer exactly. Under the mirror image rule, any attempt to accept with different terms constitutes a counteroffer. Consideration is the bargained-for exchange — something of legal value given by each party. Past consideration is not valid. Mutual assent means both parties genuinely agreed to the same terms. Contracts formed under duress, undue influence, misrepresentation, or mistake may be voidable. Written contracts are required for real estate transactions, agreements lasting more than one year, and sale of goods over $500 under the Statute of Frauds.""",
    },
    {
        "id": "doc_002",
        "topic": "Non-Disclosure Agreements (NDAs)",
        "text": """A Non-Disclosure Agreement (NDA) is a legally binding contract that restricts parties from disclosing confidential information. There are two types: unilateral NDAs (one party discloses) and mutual NDAs (both parties exchange information). Key clauses include: Definition of Confidential Information, Exclusions (public domain info, independently developed info), Duration, and Permitted Disclosures. Breach can result in injunctive relief, monetary damages, and disgorgement of profits. NDAs cannot protect general knowledge or prevent whistleblowing to government regulators.""",
    },
    {
        "id": "doc_003",
        "topic": "Employment Termination Law",
        "text": """Employment termination law governs when an employer may end an employment relationship. In most U.S. states, employment is at-will by default. Wrongful termination occurs when an employee is fired for an illegal reason including protected characteristics under Title VII, the ADA, and the ADEA. The WARN Act requires employers with 100+ employees to provide 60 days written notice before a mass layoff. Severance pay is not legally required under federal law unless promised. Employees over 40 must be given 21 days to consider a release and 7 days to revoke after signing under the Older Workers Benefit Protection Act.""",
    },
    {
        "id": "doc_004",
        "topic": "Intellectual Property — Copyright Basics",
        "text": """Copyright grants the creator of an original work exclusive rights to reproduce, distribute, display, perform, and create derivative works. Protection attaches automatically upon creation. Registration with the U.S. Copyright Office enables statutory damages up to $150,000 per work for willful infringement. Copyright protects original expression, not ideas, facts, or systems. For works created after January 1, 1978, copyright lasts for the life of the author plus 70 years. For works made for hire, copyright lasts 95 years from publication or 120 years from creation. Fair use is a defense based on four factors: purpose, nature of work, amount used, and market effect.""",
    },
    {
        "id": "doc_005",
        "topic": "Intellectual Property — Trademark Law",
        "text": """A trademark is any word, name, symbol, or logo used to identify the source of goods or services. Trademark rights arise from use in commerce. Federal registration with the USPTO provides nationwide constructive notice. Fanciful and arbitrary marks receive the strongest protection. Descriptive marks require acquired distinctiveness. Generic terms can never be registered. Trademark infringement occurs when a mark causes consumer confusion. Trademark dilution protects famous marks from blurring or tarnishment even without confusion. Section 8 Declarations must be filed between years 5–6 and every 10 years thereafter.""",
    },
    {
        "id": "doc_006",
        "topic": "Civil Litigation Process",
        "text": """Civil litigation resolves private disputes through courts. The process begins with pre-litigation investigation and litigation holds to prevent spoliation. The plaintiff files a complaint and the defendant must be served. In federal court, the defendant has 21 days to respond. Discovery includes interrogatories, document requests, depositions, and admissions. Summary judgment may be sought after discovery if there is no genuine dispute of material fact. If denied, the case proceeds to jury or bench trial. A notice of appeal must be filed within 30 days of final judgment in federal court.""",
    },
    {
        "id": "doc_007",
        "topic": "Filing Deadlines and Statutes of Limitations",
        "text": """A statute of limitations is the deadline by which a lawsuit must be filed. Missing it permanently bars the claim regardless of merits. Common limitations: personal injury 2–3 years, breach of written contract 4–6 years, breach of oral contract 2–4 years, fraud 3–6 years from discovery, defamation 1–3 years, medical malpractice 2–3 years. The clock begins when the cause of action accrues. The discovery rule tolls the statute until the plaintiff knew or should have known of the injury. Key federal deadlines: 21 days to answer a complaint, 28 days for post-trial motions, 30 days to file a notice of appeal.""",
    },
    {
        "id": "doc_008",
        "topic": "Evidence Rules and Admissibility",
        "text": """The Federal Rules of Evidence govern what evidence may be admitted in federal court. Evidence must be relevant — making a fact of consequence more or less probable. Relevant evidence may be excluded if its probative value is substantially outweighed by unfair prejudice under Rule 403. Hearsay is an out-of-court statement offered for the truth of the matter asserted and is generally inadmissible. Common exceptions include excited utterance, business records, and dying declarations. Admissions by a party-opponent are not hearsay. Expert testimony is admissible under the Daubert standard if based on sufficient facts and reliable methods reliably applied.""",
    },
    {
        "id": "doc_009",
        "topic": "Bail and Pretrial Detention",
        "text": """Bail is the temporary release of a defendant conditioned on assurance of future appearance. The Eighth Amendment prohibits excessive bail. Courts consider seriousness of charge, criminal history, community ties, flight risk, and danger to community. Types of bail include cash bail, surety bond (bondsman fee typically 10–15%), release on recognizance (ROR), and unsecured bond. Under the Bail Reform Act of 1984, federal courts may order pretrial detention if no release condition would assure appearance and community safety. Detention hearings must be held within 3 business days for flight risk or 5 days for dangerousness. United States v. Salerno (1987) upheld preventive detention as constitutional.""",
    },
    {
        "id": "doc_010",
        "topic": "Legal Ethics and Attorney-Client Privilege",
        "text": """Attorney-client privilege protects confidential communications between attorney and client made for the purpose of seeking or providing legal advice. The privilege belongs to the client — only the client can waive it. Requirements: attorney-client relationship, confidential communication, purpose of legal advice, and no waiver. The crime-fraud exception pierces privilege when the client sought help to commit a crime or fraud. The work product doctrine protects materials prepared in anticipation of litigation. Opinion work product receives near-absolute protection. Model Rules Rule 1.6 prohibits revealing client confidences without informed consent.""",
    },
]

# ── ChromaDB ───────────────────────────────────────────────
client = chromadb.Client()
try:
    client.delete_collection("capstone_kb")
except:
    pass
collection = client.create_collection("capstone_kb")
texts = [d["text"] for d in DOCUMENTS]
collection.add(
    documents=texts,
    embeddings=embedder.encode(texts).tolist(),
    ids=[d["id"] for d in DOCUMENTS],
    metadatas=[{"topic": d["topic"]} for d in DOCUMENTS],
)


# ── State ──────────────────────────────────────────────────
class CapstoneState(TypedDict):
    question: str
    messages: List[dict]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int


# ── Nodes ──────────────────────────────────────────────────
def memory_node(state):
    msgs = state.get("messages", [])
    msgs = msgs + [{"role": "user", "content": state["question"]}]
    if len(msgs) > 6:
        msgs = msgs[-6:]
    return {"messages": msgs}


def router_node(state):
    question = state["question"]
    messages = state.get("messages", [])
    recent = (
        "; ".join(f"{m['role']}: {m['content'][:60]}" for m in messages[-3:-1])
        or "none"
    )
    prompt = f"""You are a router for a Legal Document Assistant used by paralegals and lawyers.
Available routes:
- retrieve: search the legal knowledge base
- memory_only: answer from conversation history (e.g. 'what did you just say?')
- tool: use datetime tool for questions about today's date or filing deadlines
Recent conversation: {recent}
Current question: {question}
Reply with ONLY one word: retrieve / memory_only / tool"""
    decision = llm.invoke(prompt).content.strip().lower()
    if "memory" in decision:
        decision = "memory_only"
    elif "tool" in decision:
        decision = "tool"
    else:
        decision = "retrieve"
    return {"route": decision}


def retrieval_node(state):
    q_emb = embedder.encode([state["question"]]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=3)
    chunks = results["documents"][0]
    topics = [m["topic"] for m in results["metadatas"][0]]
    context = "\n\n---\n\n".join(
        f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks))
    )
    return {"retrieved": context, "sources": topics}


def skip_retrieval_node(state):
    return {"retrieved": "", "sources": []}


def tool_node(state):
    today = date.today()
    deadlines = {
        "Answer to complaint (federal)": today + timedelta(days=21),
        "Notice of appeal (federal)": today + timedelta(days=30),
        "Post-trial motions (federal)": today + timedelta(days=28),
        "Personal injury SOL (2-year)": today + timedelta(days=730),
        "Breach of contract SOL (4-year)": today + timedelta(days=1460),
    }
    lines = [f"Today's date: {today.strftime('%B %d, %Y')} ({today.isoformat()})"]
    lines.append("\nCommon filing deadlines from today:")
    for label, deadline in deadlines.items():
        lines.append(
            f"  • {label}: {deadline.strftime('%B %d, %Y')} ({(deadline-today).days} days from today)"
        )
    return {"tool_result": "\n".join(lines)}


def answer_node(state):
    question = state["question"]
    retrieved = state.get("retrieved", "")
    tool_result = state.get("tool_result", "")
    messages = state.get("messages", [])
    eval_retries = state.get("eval_retries", 0)
    context_parts = []
    if retrieved:
        context_parts.append(f"KNOWLEDGE BASE:\n{retrieved}")
    if tool_result:
        context_parts.append(f"TODAY'S DATE & DEADLINES:\n{tool_result}")
    context = "\n\n".join(context_parts)
    if context:
        system_content = f"""You are a Legal Document Assistant helping paralegals and junior lawyers.
Answer using ONLY the information provided in the context below.
If the answer is not in the context, say: I don't have that information in my knowledge base.
Do NOT add information from your training data.\n\n{context}"""
    else:
        system_content = "You are a Legal Document Assistant. Answer based on the conversation history."
    if eval_retries > 0:
        system_content += "\n\nIMPORTANT: Answer using ONLY information explicitly stated in the context above."
    lc_msgs = [SystemMessage(content=system_content)]
    for msg in messages[:-1]:
        lc_msgs.append(
            HumanMessage(content=msg["content"])
            if msg["role"] == "user"
            else AIMessage(content=msg["content"])
        )
    lc_msgs.append(HumanMessage(content=question))
    return {"answer": llm.invoke(lc_msgs).content}


def eval_node(state):
    answer = state.get("answer", "")
    context = state.get("retrieved", "")[:500]
    retries = state.get("eval_retries", 0)
    if not context:
        return {"faithfulness": 1.0, "eval_retries": retries + 1}
    prompt = f"""Rate faithfulness: does this answer use ONLY information from the context?
Reply with ONLY a number between 0.0 and 1.0.
Context: {context}
Answer: {answer[:300]}"""
    result = llm.invoke(prompt).content.strip()
    try:
        score = float(result.split()[0].replace(",", "."))
        score = max(0.0, min(1.0, score))
    except:
        score = 0.5
    return {"faithfulness": score, "eval_retries": retries + 1}


def save_node(state):
    messages = state.get("messages", [])
    messages = messages + [{"role": "assistant", "content": state["answer"]}]
    return {"messages": messages}


# ── Graph ──────────────────────────────────────────────────
def route_decision(state):
    route = state.get("route", "retrieve")
    if route == "tool":
        return "tool"
    if route == "memory_only":
        return "skip"
    return "retrieve"


def eval_decision(state):
    score = state.get("faithfulness", 1.0)
    retries = state.get("eval_retries", 0)
    if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
        return "save"
    return "answer"


g = StateGraph(CapstoneState)
g.add_node("memory", memory_node)
g.add_node("router", router_node)
g.add_node("retrieve", retrieval_node)
g.add_node("skip", skip_retrieval_node)
g.add_node("tool", tool_node)
g.add_node("answer", answer_node)
g.add_node("eval", eval_node)
g.add_node("save", save_node)
g.set_entry_point("memory")
g.add_edge("memory", "router")
g.add_conditional_edges(
    "router", route_decision, {"retrieve": "retrieve", "skip": "skip", "tool": "tool"}
)
g.add_edge("retrieve", "answer")
g.add_edge("skip", "answer")
g.add_edge("tool", "answer")
g.add_edge("answer", "eval")
g.add_conditional_edges("eval", eval_decision, {"answer": "answer", "save": "save"})
g.add_edge("save", END)

app = g.compile(checkpointer=MemorySaver())
print("✅ agent.py loaded — Legal Document Assistant ready")
print(f"   Knowledge base: {collection.count()} documents")
