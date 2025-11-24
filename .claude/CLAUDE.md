# High-Level Workflow
- When I ask a question, answer in 1-2 sentences unless I request more detail. This limit does NOT apply when generating code.
    - If you are able to answer a question in 1-2 words (e.g. "yes"), do that. I will ask for more detail if I want it.
- Do not take on tasks outside of what I ask you to do. If you are uncertain about what is in/out of scope you can ask me for confirmation. Operate on a principal of least surprise: if I review your changes and see that you changed code that I was not expecting you to change or performed a task that I did not ask you to do, that is very bad and diminishes my trust in you. Communicate proactively to avoid misunderstandings.

# Testing
- If you need to mock something when writing unit tests, use unittest.mock.patch rather than the pytest-mock library.
- Aside from the specific case detailed in the previous bullet, stick to testing with pytest whenever possible. Unittest is only used for mocking.
- Never use `pytest.mark.skip` or remove failing tests as a way to make the test suite pass. Failing tests are useful because they inform me of behavior that needs to be fixed - if you remove this, bugs can silently persist.

# Formatting
- Always leave 2 empty lines between functions. The linter will fail otherwise.

# Philosophy
- Commit changes to git frequently, in logical chunks. I will often want to pick and choose pieces to keep and pieces to rollback.
- Codebase bloat has a cost (maintenance, finding the right information, etc). Conceptually, imagine you are operating under a parsimony regularizer. For example, if you add ten tests for a trivial util, that's a lot of code to search through and maintain. If we make a trivial change later like modifying the util's error message and that breaks multiple tests, there is a time cost to fixing that.

