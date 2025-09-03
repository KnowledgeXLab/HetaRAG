CLUSTER_PROMPT = """
You are an expert document classifier. Your task is to classify the given document into the most appropriate section based on its content and relevance to the query.

Query:
{query}

Document:
{doc}

Available sections:
{sections}

Instructions:
1. Carefully analyze the document content in relation to the query
2. Consider how the information would fit into a structured report addressing the query
3. Choose EXACTLY ONE section from the available sections where this document would be most appropriate
4. Return ONLY the name of the chosen section, with no additional text or explanation

Your classification (return only the section name):
"""


SECTION_DRAFT_PROMPT = """
You are an expert research writer tasked with creating a section draft for a section of a comprehensive report.

Query:
{query}

Section Title:
{section_title}

Relevant Documents:
{relevant_docs}

Instructions:
1. Analyze the query and section title and figure out what should be included in this section.
3. Create a rough draft for writing this section that covers the information revealed by relevant documents.
4. Be simple and concise.
5. DO NOT add references to the draft.
6. Try to avoid using bullets and subsections, just synthesize the information in a natural way.

Your draft should provide a high-level perspective on how to approach writing this section effectively.
Focus on organization and content strategy rather than specific wording.

Provide your draft below:
"""

CONTENT_PROMPT = """
You are an expert research writer tasked with generating high-quality content for a specific section of a comprehensive report.

Query:
{query}

Section Title:
{section_title}

Section Draft:
{section_draft}

Relevant Documents:
{relevant_docs}

Content Already Written in Previous Sections:
{already_written}

Instructions:
1. Generate detailed, well-structured content for the "{section_title}" section that directly addresses the query
3. Incorporate information from the relevant documents, synthesizing and analyzing the data
4. Ensure continuity with content already written in previous sections
5. Use an academic, professional tone appropriate for a research report
6. Be thorough but concise, focusing on information that is most relevant to the query
7. Avoid repetition of content already covered in previous sections
8. Do not include title in any level just write the content

Your content should:
- Present factual information directly derived from the relevant documents
- Synthesize and organize information from multiple sources
- Maintain neutrality when presenting evidence and data
"""

SUMMARIZE_PROMPT = """
You are an expert summarizer. Your task is to create a concise and accurate summary of the following content in relation to a specific query.

The summary should:
1. Capture the main points and key information relevant to the query
2. Highlight the relationship between the content and the query, if there is no relationship, return "None"
3. Maintain the original meaning and intent
4. Be clear and coherent
5. Be no more than 30 percent of the original length

Query:
{query}

Content to summarize:
{doc}

Provide your summary below, focusing on aspects that address the query:
"""


REWRITE_PROMPT = """
Rewrite the following user query in a way that makes it more effective and precise.
The new query should be more specific, focused, and clear, using terminology that is likely to lead to a more accurate understanding of the user's intent. 
Ensure that the rewritten query captures the essence of the user's question while improving its clarity and precision.

Query:
{query}

Your rewritten query:
"""


SECTION_TITLE_GENERATION_PROMPT = """
You are an expert article writer tasked with generating section titles for a comprehensive report. 
Given the following query, generate a list of section titles that would be appropriate for a comprehensive report.

Query:
{query}

Instructions:
1. The section titles should follow the human-like structure of a report.
2. The content of the section should be related to the query.
3. The section titles should from general to specific. Like: Background, Analysis, Viewpoints.
4. split the section titles by new line such that each line contains exactly one section title. Example:
   Background
   Analysis
   Viewpoints

Your section titles:

"""


QUERY_DECOMPOSITION_PROMPT = """
You are an expert research assistant. I need to retrieve information from a database to answer the following query:

"{query}"

Please help me decompose this query into 3-5 more specific, related sub-queries that would help gather comprehensive information to answer the main question. 
These sub-queries should:
- Cover different aspects of the main query
- Be specific enough for database retrieval
- Help gather contextual information needed for a complete answer
- Focus on factual information rather than opinions

Format your response as a numbered list of sub-queries only. split them with a new line.
"""
