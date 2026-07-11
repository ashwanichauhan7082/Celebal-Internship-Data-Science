import time
import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt

from rag_pipeline import RAGPipeline
from hallucination_guard import check_grounding

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# List of 10 benchmark developer questions
BENCHMARK_QUESTIONS: List[str] = [
    "What is FastAPI?",
    "How does APIRouter work?",
    "Explain Dependency Injection",
    "Where is middleware handled?",
    "Path parameters routing?",
    "Why does FastAPI use Pydantic?",
    "How is Starlette related to FastAPI?",
    "What is the purpose of lifespan events?",
    "How do you define background tasks in FastAPI?",
    "Does FastAPI support WebSocket connections?"
]

def run_offline_evaluation(pipeline: RAGPipeline) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Runs the 10 benchmark questions through the RAG pipeline.
    Measures: Has Answer, Has Sources, Grounding Score, Overall Pass.
    Returns:
        results_df (pd.DataFrame): DataFrame containing metrics for each question.
        fig (plt.Figure): Matplotlib Figure visualizing the grounding scores.
    """
    logger.info("Starting offline evaluation on 10 benchmark questions...")
    results = []
    
    # Ensure vector store and chain are initialized
    if not pipeline.qa_chain:
        pipeline.setup_qa_chain()
        
    fallback_phrase = "I couldn't find this in repository history."
    
    for idx, question in enumerate(BENCHMARK_QUESTIONS, 1):
        logger.info(f"Evaluating Question {idx}/10: '{question}'")
        
        # Run query
        res = pipeline.query(question)
        answer = res["answer"]
        source_docs = res["source_documents"]
        exec_time = res["execution_time"]
        
        # 1. Has Answer (1 if we got a real response, 0 if fallback or error)
        has_answer = 0 if (fallback_phrase in answer or "Error executing query" in answer) else 1
        
        # 2. Has Sources (1 if sources found, 0 otherwise)
        has_sources = 1 if len(source_docs) > 0 else 0
        
        # 3. Grounding Score & Pass Status
        source_contents = [doc.page_content for doc in source_docs]
        is_grounded, grounding_score, verdict = check_grounding(answer, source_contents)
        
        # 4. Overall Pass (1 if grounded and has sources, or if it correctly said it couldn't find the answer)
        # Note: If it correctly said "I couldn't find this in repository history." when no sources are relevant, that is a pass!
        # Let's say it passes if grounding score is high (Safe) and it is grounded.
        overall_pass = 1 if (is_grounded and (has_sources or answer == fallback_phrase)) else 0
        
        results.append({
            "Question ID": f"Q{idx}",
            "Question": question,
            "Answer": answer[:120] + "..." if len(answer) > 120 else answer,
            "Execution Time (s)": round(exec_time, 2),
            "Has Answer": has_answer,
            "Has Sources": has_sources,
            "Grounding Score": round(grounding_score, 2),
            "Passed": overall_pass
        })
        
    df = pd.DataFrame(results)
    
    # Generate Matplotlib Bar Chart
    logger.info("Generating evaluation visualization...")
    
    # Apply modern styling
    plt.rcParams['text.color'] = '#E0E0E0'
    plt.rcParams['axes.labelcolor'] = '#E0E0E0'
    plt.rcParams['xtick.color'] = '#E0E0E0'
    plt.rcParams['ytick.color'] = '#E0E0E0'
    
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0E1117')
    ax.set_facecolor('#1A1C24')
    
    # Plot Grounding Score and Execution Time
    bars = ax.bar(df["Question ID"], df["Grounding Score"], color='#1E88E5', edgecolor='#1565C0', width=0.5, label="Grounding Score")
    
    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.0, 
            yval + 0.02, 
            f"{yval:.2f}", 
            ha='center', 
            va='bottom', 
            fontsize=9,
            color='#E0E0E0',
            weight='bold'
        )
        
    # Formatting
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Grounding Score (0.0 - 1.0)", fontsize=11, fontweight='bold', labelpad=10)
    ax.set_xlabel("Benchmark Questions", fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title("Grounding Evaluation Scores per Benchmark Question", fontsize=13, fontweight='bold', pad=15, color='#FFFFFF')
    
    # Spines styling
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#444444')
        
    ax.grid(axis='y', linestyle='--', alpha=0.2, color='#888888')
    
    # Save the chart to disk
    plt.tight_layout()
    chart_path = "evaluation_results.png"
    fig.savefig(chart_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    logger.info(f"Saved evaluation chart to {chart_path}")
    
    return df, fig

if __name__ == "__main__":
    # Test evaluation module locally if run directly
    pipeline = RAGPipeline()
    pipeline.get_or_create_vectorstore()
    pipeline.setup_qa_chain()
    
    df, fig = run_offline_evaluation(pipeline)
    print("\n--- Evaluation Results ---")
    print(df.to_string(index=False))
    print(f"\nAverage Grounding Score: {df['Grounding Score'].mean():.4f}")
    print(f"Overall Pass Rate: {df['Passed'].mean() * 100:.1f}%")
