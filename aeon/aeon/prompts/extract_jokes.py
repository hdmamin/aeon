from pydantic import BaseModel, Field


class Response(BaseModel):
    joke: str = Field(
        ...,
        description="One or more sentences containing a joke from the input text. In most cases "
                    "this should quote the source verbatim, but you may apply minor cleanup to "
                    "remove transcription artifacts, filler words like 'uh', etc. "
                    "The transcripts are from live spoken performances but this field is "
                    "designed to be *read*."
    )

    prompt: str = Field(
        ...,
        description="A one line comment or question that you imagine could have elicited this "
                    "joke in conversation, from the comic's interlocutor."
    )
        
    subtext: str = Field(
        ...,
        description="The often banal observation underlying the joke. This should not be funny."
    )
        
    joke_variant: str = Field(
        ...,
        description="Your attempt to modify the original joke while maintaining the same subtext. "
                    "Unlike the subtext, this will typically "
    )


class BatchResponse(BaseModel):
    items: list[Response]


# TODO: maybe switch examples to a less toilety joke
# TODO: add unfunny variant (currently does not match schema)
# TODO: consider adding good variants? (Recall claude ciabatta jokes were decent, but if labeling
# with cheaper models this likely won't work well)
messages = [
    {
        "role": "developer",
        "content": """
<instructions>
You are a detail-oriented research assistant with a shrewd understanding of humor, human behavior, and writing. You are performing one step in a data pipeline for a humor-related research project. Your task is to extract all jokes from the input passage and return valid JSON where each object contains "joke" and "subtext" fields. You can think of the subtext as the often banal point that the joke is making, which the comedian has ultimately massaged or restructured such that the resulting joke subverts the audience's expectations in a fun way. The subtext should not be funny, e.g. it might plainly state "airplane food is bad".

Not every sentence in the input must belong to a joke, but most of them probably will because I am showing you standup transcripts. Some of the transcripts will be very long and so your response may contain many items: potentially up to hundreds of jokes. Extracting a joke is not an endorsement of the viewpoint that joke expresses.
    
<example_input>
[comedian: Louis CK]
I could never wear white pants because I’ll get my period, first of all. I know that. [Laughter] Or diarrhea, more likely. [Laughter] Which is — That’s really my period. Diarrhea. About once a month, I’m like, “Oh, fuck, here we go.” [Laughter] “Better just get home. And don’t make any big decisions today.” [Laughter] It’s true. Don’t make — You know, if you have diarrhea, don’t, like, negotiate. It’s a bad bargaining position. If I have diarrhea, you stand between me and the toilet, I’ll sell you my house for 10 cents.
</example_input>
<example_output>
[
    {
        "joke": "I could never wear white pants because I’ll get my period, first of all. I know that. Or diarrhea, more likely. That’s really my period: diarrhea.",
        "prompt": "I've been thinking of getting white pants.",
        "subtext": "Wearing white pants makes any bodily mishap immediately obvious, which is risky if your digestive issues are routine and unavoidable."
        "joke_variant": "TODO"
    },
    {
        "joke": "About once a month, I’m like, “Oh, fuck, here we go. “Better just get home. And don’t make any big decisions today.” It’s true. If you have diarrhea, don’t, like, negotiate. It’s a bad bargaining position. If I have diarrhea, you stand between me and the toilet, I’ll sell you my house for 10 cents.",
        "subtext": "Being desperate for a bathroom gives you zero leverage in any negotiation."
        "joke_variant": "TODO"
    }
]

[
    {
        'joke': 'The past couple years, I’ve done a lot of work on myself. And I’ve realized that I’ll be fine as long as I get constant attention.',
        'subtext': 'People often seek attention to feel okay about themselves.',
        'joke_variant': "I've learned that I need constant attention to feel okay."
    },
    {
    'joke': 'When I was three years old, they pulled me aside and they told me that I was adopted. And that my real mother had been murdered… by Miss America.',
    'subtext': 'Children can misunderstand complex family situations in humorous ways.',
    'joke_variant': "As a child, I misunderstood my family's explanations about my adoption and my mother's death."},
    {'joke': 'They thought of every angle.',
    'subtext': 'People often prepare for all possible questions or reactions.',
    'joke_variant': 'They considered every possible way I might react or ask questions.'},
    {'joke': 'And to compound the stress that I was under, when I was three years old, I thought that Miss America was the Statue of Liberty.',
    'subtext': 'Children can confuse public figures or symbols with other objects or concepts.',
    'joke_variant': 'As a child, I confused Miss America with the Statue of Liberty.'},
    {'joke': 'And that summer was 1986. In ’86 that summer on the Fourth of July, Reagan was president then, and he gave a speech in front of the Statue of Liberty. He was wearing a white shirt, the wind was blowing, and he goes… ‘This lady is a great lady.’ And I was sitting at home like, ‘Oh, my God.’ ‘This goes all the way to the top.’ ‘I’m never gonna get justice for my mother.’',
    'subtext': 'Children can interpret political speeches in a literal and emotional way.',
    'joke_variant': "As a kid, I took a presidential speech literally and thought it meant I wouldn't get justice for my mother."},
    {'joke': 'I always wanted attention in school, like to a sick degree. I really… I mean, I don’t know if you guys ever had this feeling, but do you remember when you were in elementary school, grammar school, and a kid in your class would come to school one day and you’d find out their grandparent had died. And they would get, like, so much attention.',
    'subtext': 'Children often crave attention, especially during emotional moments.',
    'joke_variant': 'I craved attention in school, especially when someone else experienced a tragedy.'}
]
</example_output>
</instructions>
"""
    },
    {
        "role": "user",
        "content": "$transcript"
    }
]

kwargs = {
    "model": "gpt-4.1",
    "temperature": 0.3,
    "response_format": BatchResponse,
}