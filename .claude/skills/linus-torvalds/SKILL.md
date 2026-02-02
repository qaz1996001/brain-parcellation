---
name: linus-torvalds
description: |
  Apply Linus Torvalds' programming philosophy for code design, architecture, and planning decisions.
  Use when: (1) Designing data structures or system architecture, (2) Code review with focus on simplicity
  and readability, (3) Refactoring to eliminate special cases, (4) Planning implementation approach,
  (5) Evaluating trade-offs between theory and practicality, (6) User mentions "Linus", "Linux kernel style",
  "good taste", or requests pragmatic engineering advice.
---

# Linus Torvalds Programming Philosophy

## Core Principles

### 1. Code Over Talk
> "Talk is cheap. Show me the code."

- Prove ideas with working code, not discussion
- Patches beat complaints
- Theory loses when it clashes with practice

### 2. Data Structures First
> "Bad programmers worry about the code. Good programmers worry about data structures and their relationships."

- Design data structures before writing code
- Correct structures make algorithms obvious
- Complex code often signals wrong data design

### 3. Good Taste = Eliminate Special Cases

**Bad (CS101):**
```c
if (!prev)
    *head = entry->next;
else
    prev->next = entry->next;
```

**Good (indirect pointer):**
```c
node **p = head;
while (*p != entry)
    p = &(*p)->next;
*p = entry->next;
```

- Reframe problems so edge cases become normal cases
- Deep understanding produces elegant solutions

### 4. Spartan Style

| Rule | Guideline |
|------|-----------|
| Indentation | 8-char tabs; >3 levels = refactor needed |
| Functions | Short (~24 lines), do one thing |
| Local vars | Short names (`i`, `tmp`, `p`) |
| Global funcs | Descriptive (`count_active_users()`) |
| Braces | K&R style |
| Comments | Explain "why", not "what" |

### 5. Pragmatic Engineering
> "I'm not a visionary. I want to fix the pothole in front of me before I fall in."

- Solve immediate problems, not hypothetical ones
- Let code evolve through real usage
- Start small, iterate based on feedback

## Decision Framework

When evaluating code or design:

1. **Is the data structure right?** If code is complex, check data design first
2. **Any special cases to eliminate?** Look for different angles
3. **Within 3 levels of indentation?** If not, split functions
4. **Would 100 random programmers understand immediately?** Clarity > cleverness
5. **Solving a real problem?** Avoid premature abstraction

## Anti-Patterns to Avoid

| Don't | Do Instead |
|-------|------------|
| Weeks of design without code | Prototype early |
| Clever tricks | Clear, obvious code |
| Long functions | Short, focused functions |
| Deep nesting | Extract and simplify |
| Complex code for wrong data | Fix data structure |
| Polite acceptance of bad code | Direct, honest feedback |

## Full Philosophy Reference

For complete principles, examples, and coding style details:
→ See [references/philosophy.md](references/philosophy.md)

Sections include:
- Detailed code examples with analysis
- Linux Kernel coding style summary
- Comprehensive checklist for code review
- Extended quotes and context
