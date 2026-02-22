"""
Custom Re-ranker System:
1. Retrieve top-50 via ANN (Approximate Nearest Neighbor)
2. Re-rank top-10 using LLM semantic similarity
3. Return top-3 for answer synthesis
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import json


@dataclass
class Document:
    """Document with content and metadata"""
    id: str
    content: str
    metadata: Dict
    score: float = 0.0


class CustomReranker:
    """
    Advanced re-ranker that uses LLM for semantic similarity scoring
    """
    
    def __init__(self, vectorstore, llm):
        """
        Args:
            vectorstore: Vector database (Pinecone, Chroma, etc.)
            llm: Language model for re-ranking (OpenAI, Anthropic, etc.)
        """
        self.vectorstore = vectorstore
        self.llm = llm
    
    def retrieve(self, query: str, k: int = 50) -> List[Document]:
        """
        Step 1: Retrieve top-k documents via ANN search
        
        Args:
            query: User query
            k: Number of documents to retrieve (default 50)
            
        Returns:
            List of top-k documents with similarity scores
        """
        print(f"📊 Step 1: Retrieving top-{k} via ANN search...")
        
        # Perform vector similarity search
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        # Convert to Document objects
        documents = []
        for doc, score in results:
            documents.append(Document(
                id=doc.metadata.get('id', f'doc_{len(documents)}'),
                content=doc.page_content,
                metadata=doc.metadata,
                score=score
            ))
        
        print(f"✓ Retrieved {len(documents)} documents")
        print(f"  Score range: {min(d.score for d in documents):.3f} - {max(d.score for d in documents):.3f}")
        return documents
    
    def rerank_with_llm(self, query: str, documents: List[Document], 
                        top_n: int = 10) -> List[Document]:
        """
        Step 2: Re-rank top-N documents using LLM for precise semantic scoring
        
        Args:
            query: User query
            documents: Documents to re-rank
            top_n: Number of top documents to re-rank (default 10)
            
        Returns:
            Re-ranked documents with new scores
        """
        print(f"\n🎯 Step 2: Re-ranking top-{top_n} using LLM...")
        
        # Take only top_n for re-ranking (reduce cost)
        candidates = documents[:top_n]
        
        # Build prompt for LLM re-ranking
        rerank_prompt = self._build_rerank_prompt(query, candidates)
        
        # Get LLM to score relevance
        response = self.llm.predict(rerank_prompt)
        
        # Parse scores from LLM response
        scores = self._parse_llm_scores(response, len(candidates))
        
        # Update document scores
        for doc, new_score in zip(candidates, scores):
            doc.score = new_score
        
        # Sort by new scores (highest first)
        candidates.sort(key=lambda d: d.score, reverse=True)
        
        print(f"✓ Re-ranked {len(candidates)} documents")
        print(f"  New score range: {min(d.score for d in candidates):.3f} - {max(d.score for d in candidates):.3f}")
        
        return candidates
    
    def _build_rerank_prompt(self, query: str, documents: List[Document]) -> str:
        """Build prompt for LLM re-ranking"""
        
        docs_text = ""
        for i, doc in enumerate(documents, 1):
            docs_text += f"\n[Document {i}]\n{doc.content[:500]}...\n"
        
        prompt = f"""You are a relevance scoring system. Score each document's relevance to the query on a scale of 0-10.

Query: "{query}"

Documents:
{docs_text}

Instructions:
1. Carefully evaluate how well each document answers the query
2. Consider semantic meaning, not just keyword matching
3. Score 10 = Perfect answer, 0 = Completely irrelevant
4. Be precise and critical

Return ONLY a JSON array of scores in order, like: [8.5, 6.2, 9.1, ...]

Scores:"""
        
        return prompt
    
    def _parse_llm_scores(self, response: str, expected_count: int) -> List[float]:
        """Parse scores from LLM response"""
        try:
            # Try to extract JSON array
            response = response.strip()
            if response.startswith('['):
                scores = json.loads(response)
            else:
                # Try to find JSON in response
                import re
                match = re.search(r'\[([\d\.,\s]+)\]', response)
                if match:
                    scores = json.loads(match.group(0))
                else:
                    raise ValueError("No JSON array found")
            
            # Validate
            if len(scores) != expected_count:
                print(f"⚠️  Warning: Expected {expected_count} scores, got {len(scores)}")
                scores = scores[:expected_count] + [5.0] * (expected_count - len(scores))
            
            # Normalize to 0-1 range
            scores = [float(s) / 10.0 for s in scores]
            return scores
            
        except Exception as e:
            print(f"⚠️  Error parsing scores: {e}. Using fallback scores.")
            # Fallback: linear decay
            return [1.0 - (i / expected_count) for i in range(expected_count)]
    
    def get_top_k(self, documents: List[Document], k: int = 3) -> List[Document]:
        """
        Step 3: Return top-k documents for answer synthesis
        
        Args:
            documents: Re-ranked documents
            k: Number of top documents to return (default 3)
            
        Returns:
            Top-k documents
        """
        print(f"\n📝 Step 3: Selecting top-{k} for answer synthesis...")
        
        top_docs = documents[:k]
        
        print(f"✓ Selected {len(top_docs)} documents:")
        for i, doc in enumerate(top_docs, 1):
            print(f"  {i}. Score: {doc.score:.3f} | ID: {doc.id}")
        
        return top_docs
    
    def search_and_rerank(self, query: str, 
                          retrieve_k: int = 50,
                          rerank_k: int = 10, 
                          final_k: int = 3) -> List[Document]:
        """
        Complete pipeline: Retrieve -> Re-rank -> Return top docs
        
        Args:
            query: User query
            retrieve_k: Number to retrieve via ANN (default 50)
            rerank_k: Number to re-rank with LLM (default 10)
            final_k: Number to return for synthesis (default 3)
            
        Returns:
            Top final_k documents after re-ranking
        """
        print(f"\n{'='*70}")
        print(f"🔍 CUSTOM RE-RANKING PIPELINE")
        print(f"{'='*70}")
        print(f"Query: \"{query}\"\n")
        
        # Step 1: ANN retrieval
        documents = self.retrieve(query, k=retrieve_k)
        
        # Step 2: LLM re-ranking
        reranked = self.rerank_with_llm(query, documents, top_n=rerank_k)
        
        # Step 3: Get top-k for synthesis
        final_docs = self.get_top_k(reranked, k=final_k)
        
        print(f"\n{'='*70}")
        print(f"✅ PIPELINE COMPLETE")
        print(f"{'='*70}\n")
        
        return final_docs


# ============================================================================
# EXAMPLE USAGE WITH MOCK DATA
# ============================================================================

class MockVectorStore:
    """Mock vector store for demonstration"""
    
    def __init__(self):
        # Sample documents about Python programming
        self.documents = [
            ("Python lists are mutable sequences that can store multiple items.", 
             {"id": "doc_1", "topic": "lists"}),
            ("List comprehensions provide a concise way to create lists in Python.", 
             {"id": "doc_2", "topic": "lists"}),
            ("Dictionaries in Python store key-value pairs.", 
             {"id": "doc_3", "topic": "dicts"}),
            ("Python functions are defined using the def keyword.", 
             {"id": "doc_4", "topic": "functions"}),
            ("Lambda functions in Python are anonymous functions.", 
             {"id": "doc_5", "topic": "functions"}),
            ("Python classes support object-oriented programming.", 
             {"id": "doc_6", "topic": "oop"}),
            ("List methods like append, extend, and insert modify lists in place.", 
             {"id": "doc_7", "topic": "lists"}),
            ("Python decorators modify function behavior.", 
             {"id": "doc_8", "topic": "decorators"}),
            ("Generators in Python use yield to produce values lazily.", 
             {"id": "doc_9", "topic": "generators"}),
            ("Python's enumerate function returns index-value pairs.", 
             {"id": "doc_10", "topic": "builtins"}),
        ] * 5  # Duplicate to simulate 50 docs
    
    def similarity_search_with_score(self, query: str, k: int):
        """Simulate vector similarity search"""
        from langchain.schema import Document
        
        # Simple scoring based on keyword overlap (mock)
        results = []
        for content, metadata in self.documents[:k]:
            # Mock similarity score (0-1)
            score = 0.5 + (hash(content + query) % 100) / 200.0
            results.append((Document(page_content=content, metadata=metadata), score))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]


class MockLLM:
    """Mock LLM for demonstration"""
    
    def predict(self, prompt: str) -> str:
        """Simulate LLM scoring"""
        # Extract number of documents
        import re
        doc_count = len(re.findall(r'\[Document \d+\]', prompt))
        
        # Generate realistic-looking scores
        scores = []
        for i in range(doc_count):
            # Simulate varying relevance (higher scores for early docs)
            base_score = 9.0 - (i * 0.5)
            variation = (hash(str(i)) % 20 - 10) / 10.0
            score = max(1.0, min(10.0, base_score + variation))
            scores.append(round(score, 1))
        
        return json.dumps(scores)


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CUSTOM RE-RANKER DEMO")
    print("="*70)
    
    # Initialize mock components
    vectorstore = MockVectorStore()
    llm = MockLLM()
    
    # Create re-ranker
    reranker = CustomReranker(vectorstore, llm)
    
    # Test query
    query = "How do I modify Python lists?"
    
    # Run complete pipeline
    final_docs = reranker.search_and_rerank(
        query=query,
        retrieve_k=50,   # Retrieve 50 via ANN
        rerank_k=10,     # Re-rank top 10 with LLM
        final_k=3        # Return top 3
    )
    
    # Display final results
    print("\n" + "="*70)
    print("FINAL TOP-3 DOCUMENTS FOR ANSWER SYNTHESIS")
    print("="*70)
    
    for i, doc in enumerate(final_docs, 1):
        print(f"\n[Document {i}] (Score: {doc.score:.3f})")
        print(f"ID: {doc.id}")
        print(f"Content: {doc.content}")
        print(f"Metadata: {doc.metadata}")
    
    print("\n" + "="*70)
    print("✅ Re-ranking complete! These top-3 docs can now be used for RAG.")
    print("="*70)
    
    # Show how to use with real LangChain components
    print("\n" + "="*70)
    print("INTEGRATION WITH REAL SYSTEMS")
    print("="*70)
    
    integration_code = """
# Integration with real LangChain components:

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
import pinecone

# Initialize real components
embeddings = OpenAIEmbeddings()
vectorstore = Pinecone.from_existing_index("my-index", embeddings)
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Create reranker with real components
reranker = CustomReranker(vectorstore, llm)

# Use in RAG pipeline
query = "What are the company's risk factors?"
top_docs = reranker.search_and_rerank(
    query=query,
    retrieve_k=50,   # ANN search
    rerank_k=10,     # LLM re-rank
    final_k=3        # Final synthesis
)

# Generate answer using top-3 docs
context = "\\n\\n".join([doc.content for doc in top_docs])
answer = llm.predict(f"Context: {context}\\n\\nQuestion: {query}\\n\\nAnswer:")
"""
    
    print(integration_code)