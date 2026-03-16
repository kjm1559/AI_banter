# AI Banter Project Code of Conduct

## Development Principles

### 1. Plan-Driven Development
- All development work must begin with a **concrete plan**
- Manage the plan using a Todo list and track the status of each step in real-time
- Do not start implementation without a plan
- Immediately update the plan if the scope of work changes

### 2. Modular Testing
- Each feature module must have **independent tests**
- Write tests alongside module development
- Tests must accurately reflect the intended purpose of the module
- Isolate tests that depend on other modules using mocking

### 3. Test Completion Standard
- Development is considered complete only when **all tests pass successfully**
- Test coverage must include all major use cases for the feature
- Error handling paths must also be covered
- No build or lint errors are allowed

### 4. Atomic Commit Strategy
- Create a **separate commit for each completed module**
- Commit messages must clearly describe the changes and their intent
- Push to the remote repository immediately after committing
- Do not bundle multiple modules into a single commit

## Workflow

1. **Requirements Analysis** → Plan Creation (Todo Generation)
2. **Module Design** → Decompose into Module Units
3. **Module Development** → Develop with Parallel Testing
4. **Module Verification** → Test Passing, Lint Check
5. **Commit & Push** → Execute Immediately Upon Module Completion
6. **All Modules Completed** → Verify All Tests Pass
7. **Development Complete** → All Tests and Build Success

## Verification Checklist

- [ ] Plan (Todo) created with all steps defined
- [ ] Each module has independent tests
- [ ] All tests pass
- [ ] No build or lint errors
- [ ] Commit and push completed for each module
- [ ] All modules developed and full functionality working

---

*This code of conduct is established to maintain a consistent development process and ensure quality.*
