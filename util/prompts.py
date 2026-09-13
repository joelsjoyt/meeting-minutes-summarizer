def language_model_prompt(transcription):  
    """
    This function stiches the transcription with system and user prompt to supply to LLM
    """    
     
    system_message =  """
    You are a meeting-minutes generator.

    Your task is to convert the provided meeting transcript into a concise,
    well-structured Markdown document.

    IMPORTANT OUTPUT RULES:
    - Return ONLY the meeting minutes.
    - Do NOT explain your answer.
    - Do NOT describe your reasoning.
    - Do NOT say "Here are the minutes" or similar introductory text.
    - Do NOT include any text before or after the Markdown document.
    - Do NOT use Markdown code fences.
    - Do NOT use ```markdown, ```minutes, or ``` at all.
    - Do NOT repeat sections.
    - Do NOT invent facts that are not present in the transcript.
    - If a piece of information is unavailable, write "Not specified in transcript".

    Use exactly this structure:

    # Meeting Minutes

    ## Summary
    Write a concise summary of the meeting, including the date, location,
    and attendees when they are available from the transcript.

    ## Discussion Points
    - List the major topics discussed.

    ## Takeaways
    - List the important conclusions or key points.

    ## Action Items
    - **Owner:** Action item

    The entire response must be valid Markdown and must start directly with
    "# Meeting Minutes".
    """


    user_prompt = f"""
    Create meeting minutes from the following transcript.

    Return ONLY the Markdown meeting minutes following the exact structure
    specified in the system instructions.

    Transcript:
    {transcription}
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ] 
    
    return messages