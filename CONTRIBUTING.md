# Contributing Guide For UserRL

This page lists the operational governance model of this project, as well as the recommendations and requirements for how to best contribute to UserRL. We strive to obey these as best as possible. As always, thanks for contributing – we hope these guidelines make it easier and shed some light on our approach and processes.

# Governance Model

## Salesforce Sponsored

The intent and goal of open sourcing this project is to increase the contributor and user base. However, only Salesforce employees will be given `admin` rights and will be the final arbitrars of what contributions are accepted or not.

# Getting started

Please join the community by:
- Opening issues and discussions on our [GitHub repository](https://github.com/SalesforceAIResearch/UserRL)
- Following our research updates and announcements
- Contributing to our diverse gym environments and evaluation frameworks

Also please make sure to take a look at the project documentation in the respective README files to understand our current capabilities and future directions.

# Issues, requests & ideas

Use GitHub Issues page to submit issues, enhancement requests and discuss ideas.

### Bug Reports and Fixes
-  If you find a bug, please search for it in the [Issues](https://github.com/SalesforceAIResearch/UserRL/issues), and if it isn't already tracked,
   [create a new issue](https://github.com/SalesforceAIResearch/UserRL/issues/new). Fill out the "Bug Report" section of the issue template. Even if an Issue is closed, feel free to comment and add details, it will still be reviewed.
-  Issues that have already been identified as a bug (note: able to reproduce) will be labelled `bug`.
-  If you'd like to submit a fix for a bug, [send a Pull Request](#creating_a_pull_request) and mention the Issue number.
  -  Include tests that isolate the bug and verifies that it was fixed.
  -  For gym environment bugs, ensure your fix works across all supported environments.

### New Features
-  If you'd like to add new functionality to this project, describe the problem you want to solve in a [new Issue](https://github.com/SalesforceAIResearch/UserRL/issues/new).
-  Issues that have been identified as a feature request will be labelled `enhancement`.
-  If you'd like to implement the new feature, please wait for feedback from the project
   maintainers before spending too much time writing the code. In some cases, `enhancement`s may
   not align well with the project objectives at the time.
-  For new gym environments, follow the guidelines in [gyms/README.md](gyms/README.md) and ensure compatibility with our multi-turn training framework.

### Tests, Documentation, Miscellaneous
-  If you'd like to improve the tests, you want to make the documentation clearer, you have an
   alternative implementation of something that may have advantages over the way its currently
   done, or you have any other change, we would be happy to hear about it!
  -  If its a trivial change, go ahead and [send a Pull Request](#creating_a_pull_request) with the changes you have in mind.
  -  If not, [open an Issue](https://github.com/SalesforceAIResearch/UserRL/issues/new) to discuss the idea first.
-  For RL training improvements, ensure compatibility with GRPO algorithm and multi-turn credit assignment.
-  Documentation improvements are especially welcome for gym environments, training configurations, and evaluation procedures.

If you're new to our project and looking for some way to make your first contribution, look for
Issues labelled `good first contribution`.

# Contribution Checklist

- [x] Clean, simple, well styled code following Python best practices
- [x] Commits should be atomic and messages must be descriptive. Related issues should be mentioned by Issue number.
- [x] Comments
  - Module-level & function-level comments.
  - Comments on complex blocks of code or algorithms (include references to sources).
  - For RL algorithms, document hyperparameter choices and experimental rationale.
- [x] Tests
  - The test suite, if provided, must be complete and pass
  - Increase code coverage, not versa.
  - For gym environments, include integration tests with the training pipeline.
  - For RL components, include unit tests for reward calculations and credit assignment.
- [x] Dependencies
  - Minimize number of dependencies.
  - Prefer Apache 2.0, BSD3, MIT, ISC and MPL licenses.
  - Ensure compatibility with PyTorch, Transformers, and Ray frameworks.
- [x] Reviews
  - Changes must be approved via peer code review
  - For RL training changes, include performance benchmarks and ablation studies when applicable.

# Creating a Pull Request

1. **Ensure the bug/feature was not already reported** by searching on GitHub under Issues.  If none exists, create a new issue so that other contributors can keep track of what you are trying to add/fix and offer suggestions (or let you know if there is already an effort in progress).
2. **Fork** the repository on GitHub.
3. **Clone** the forked repo to your machine.
4. **Create** a new branch to contain your work (e.g. `git checkout -b fix-issue-11`)
5. **Install** UserRL and dependencies following the installation guide in README.md
6. **Commit** changes to your own branch with descriptive messages.
7. **Push** your work back up to your fork. (e.g. `git push origin fix-issue-11`)
8. **Submit** a Pull Request against the `main` branch and refer to the issue(s) you are fixing. Try not to pollute your pull request with unintended changes. Keep it simple and small.
9. **Sign** the Salesforce CLA (you will be prompted to do so when submitting the Pull Request)

> **NOTE**: Be sure to [sync your fork](https://help.github.com/articles/syncing-a-fork/) before making a pull request.

# Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Salesforce's open source projects.

Complete your CLA here: <https://cla.salesforce.com/sign-cla>

# Issues
We use GitHub issues to track public bugs and feature requests. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue. For RL training issues,
include:
- Model configuration details
- Training hyperparameters
- Environment setup information
- Error logs and stack traces

# Code of Conduct
Please follow our [Code of Conduct](CODE_OF_CONDUCT.md).

# License
By contributing your code, you agree to license your contribution under the terms of our project [LICENSE](LICENSE.txt) and to sign the [Salesforce CLA](https://cla.salesforce.com/sign-cla)
