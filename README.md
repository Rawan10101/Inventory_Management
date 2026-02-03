# Deloitte x AUC Hackathon

## Introduction

Welcome to the Deloitte x AUC Hackathon. As potential future consultants at Deloitte, you will face a reality that defines the consulting profession: clients come to us with problems, not solutions.

### The Consultant's Challenge

Clients expect consultants to diagnose their challenges and architect innovative solutions. You will not receive a step-by-step guide, just like in the real world. Success requires both technical excellence and business acumen across four key dimensions:

**Technical Skills**: Can you build your solution?

**Business Thinking**: Should you build it? What value does it provide?

**Team Work**: Can you deliver it together?

**Communication**: Can you sell your solution?

### How to Approach This Challenge

Each use case below presents a client with a real business problem. The business questions provided are meant to help guide your thinking and give you ideas on possible features to implement. Your approach should be:

1. Frame the business problem and understand the value proposition
2. Identify which business questions your solution will address
3. Design and implement technical features that deliver measurable business impact
4. Document how your solution creates value for the client (feel free to name your proposed product something creative)

Remember: clients look for measurable impact, scalability, and competitive advantages. Think like a consultant, build like an engineer.

---

## Dataset Download

Due to file size limitations, the datasets are provided via GitHub Releases.

### Quick Start

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ynakhla/DIH-X-AUC-Hackathon.git
   cd DIH-X-AUC-Hackathon
   ```

2. **Download Dataset from Release**
   
   **[Download Datasets Here →](https://github.com/ynakhla/DIH-X-AUC-Hackathon/releases/tag/v1.0-data)**

   Choose the data for your use case:
   - **inventory-management.zip** (667 MB) - Use Case 1: Fresh Flow Markets
   - **menu-engineering-part1.zip** (1.4 GB) - Use Case 2: Flavor Flow Craft
   - **menu-engineering-part2.zip** (1.2 GB) - Use Case 2: Flavor Flow Craft
   - **shift-planning.zip** (292 MB) - Use Case 3: Quick Serve Kitchens

   **Note:** Menu Engineering requires BOTH Part 1 and Part 2

3. **Extract to Data Folder**
   
   Extract the downloaded ZIP file(s) into the `data/` directory. Your structure should be:
   ```
   DIH-X-AUC-Hackathon/
   ├── data/
   │   ├── Inventory Management/     (CSV files)
   │   ├── Menu Engineering Part 1/  (CSV files)
   │   ├── Menu Engineering Part 2/  (CSV files)
   │   └── Shift Planning/           (CSV files)
   └── src/
   ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Data Notes
- All timestamps are UNIX integers (use `FROM_UNIXTIME()` in MySQL)
- All monetary values are in DKK (Danish Krone)
- See `data/README.md` for detailed documentation

---

## Repository & Documentation Requirements

### Recommended Folder Structure

```
your-project/
├── README.md                    # Required
├── requirements.txt             # Python dependencies
├── package.json                 # Node.js dependencies
├── src/                         # Source code
│   ├── main.py                  # Entry point
│   ├── models/                  # Data models/ML models
│   ├── services/                # Business logic
│   ├── utils/                   # Utility functions
│   └── api/                     # API endpoints
├── tests/                       # Test files
├── docs/                        # Additional documentation
├── config/                      # Configuration files
└── data/                        # Sample data (no sensitive info)
```

### Required Components

### 1. README.md

Your README is the first document evaluators will review. It must include:

- **Project Name and Description**: A clear, concise description of what your project does and the problem it solves
- **Features**: List all features and functionalities with screenshots from your UI
- **Technologies Used**: Document all technologies, frameworks, and external APIs used
- **Installation**: Provide step-by-step installation instructions
- **Usage**: Include usage guidelines and examples
- **Architecture**: Brief overview of the system architecture (optional but recommended)
- **Team Members**: List all team members with their roles and contributions

#### Team Collaboration Requirements

The evaluation heavily weighs **team collaboration** based on GitHub commit history:

| Distribution | Assessment | Impact on Score |
|--------------|------------|-----------------|
| Even split (e.g., 33%/33%/34%) | Excellent | Full points |
| Reasonable split (e.g., 45%/35%/20%) | Good | Most points |
| Uneven split (e.g., 80%/15%/5%) | Concerning | Reduced points |
| Single contributor (100%) | Poor | Significant penalty |

Note: If a team member is a non technical member, please document their contributions in the README file to ensure fair assessment.

**Best Practices for Commits:**

1. Each team member should commit their own work - Don't have one person commit everything
2. Use meaningful commit messages:
   - Good: "Add user authentication with JWT tokens"
   - Bad: "update" or "fix"
3. Commit regularly - Small, frequent commits are better than one massive commit
4. Use branches for features - Merge via pull requests when possible

**Check Your Contribution Distribution:**

Run this command in your repository:
```bash
git shortlog -sn --all
```

### 2. Code Organization
- Use proper file extensions for all code files
- Maintain a clear and logical folder structure
- Separate concerns into appropriate directories (models, services, utils, api)
- Include an entry point file (e.g., main.py, index.js)

### 3. Dependencies Management
- Include a dependencies file (requirements.txt for Python, package.json for Node.js, etc.)
- List all libraries and frameworks required to run the project
- Specify version numbers where applicable

### 4. Code Quality
- Add meaningful comments and docstrings to all functions and classes
- Include file headers explaining the purpose of each module
- Follow language-specific best practices and conventions
- Ensure code runs without errors

### 5. Security
- Do not include sensitive data such as API keys, passwords, or tokens
- Use environment variables for configuration
- Include a .gitignore file to exclude sensitive or unnecessary files

### 6. Testing
- Include test files in a dedicated tests directory
- Document how to run tests

### 7. Additional Documentation
- Provide architecture overview or diagrams (optional but recommended)
- Include any additional documentation in a docs directory

### 8. Repository Access
- Repository must be public
- Use the main branch for submission
- Ensure all team members have appropriate access


---

## Use Cases (Clients)

### Fresh Flow Markets: Inventory Management

**The Challenge**

Restaurant and grocery owners face a relentless balancing act. Over-stocking leads to waste and expired inventory eating away at profits. Under-stocking causes stockouts, lost revenue, and frustrated customers. The root cause? Poor demand forecasting. Without accurate predictions, businesses are trapped in a cycle of unnecessary costs, reduced profitability, and unsustainable operations. FreshFlow needs intelligent systems, not gut instinct.

**Potential Business Questions**

The following questions are provided to help guide your thinking and inspire potential features for your solution:

- How do we accurately predict daily, weekly, and monthly demand?
- What prep quantities should kitchens prepare to minimize waste?
- How can we prioritize inventory based on expiration dates?
- What promotions or bundles can move near-expired items profitably?
- How do external factors (weather, holidays, weekends) impact sales?

---

### Flavor Flow Craft: Menu Engineering

**The Challenge**

FlavorCraft sits on a goldmine of historical sales data—every order, every customer preference—yet they're making menu decisions on hunches. They don't know which dishes are secretly losing money, or what tweaks could turn underperformers into bestsellers. This isn't just a missed opportunity; it's revenue left on the table. Your mission is to create a data-driven assistant that analyzes their menu and sales data, suggesting improvements to items, descriptions, and pricing. But don't stop there—your client manager encourages innovative thinking beyond the obvious.

**Potential Business Questions**

The following questions are provided to help guide your thinking and inspire potential features for your solution:

- Which menu items are stars, plowhorses, puzzles, or dogs?
- How should we adjust pricing to maximize profitability?
- What wording or descriptions increase item sales?
- Which items should be promoted, re-engineered, or eliminated?
- What hidden patterns exist in customer purchasing behavior?



### Quick Serve Kitchens: Shift Planning

**The Challenge**

Monday's schedule looks perfect. Wednesday: three call-offs. Friday: foot traffic doubles because TikTok made your item viral. The schedule is now a dumpster fire. This is QuickServe's weekly reality. The core problem? Accurately predicting and meeting wildly fluctuating customer demand to ensure optimal staffing on every shift. Too few people means terrible service and burnout. Too many means spiraling labor costs. QuickServe needs a Shift Wizard—an intelligent system that monitors schedules, coverage, PTO, surprise events, and constantly recommends the next best move.

**Potential Business Questions**

The following questions are provided to help guide your thinking and inspire potential features for your solution:

- How do we predict demand spikes from social media, weather, or events?
- What's the optimal staffing level for each shift?
- How do we quickly adjust when call-offs happen?
- How can we balance labor costs with service quality?
- How do we incorporate employee preferences while meeting business needs?

---

## Evaluation Criteria

### Documentation Quality
Your project documentation will be evaluated based on the clarity and completeness of your README file, the presence of code comments and docstrings, and the overall organization of documentation materials.

### Code Architecture & Modularity
The structure and organization of your codebase will be assessed, including proper separation of concerns, logical folder hierarchy, and modular design patterns that promote maintainability and scalability.

### Team Collaboration
Collaboration will be evaluated through GitHub commit history, ensuring all team members have contributed meaningfully to the project with clear and descriptive commit messages.

### AI/ML Integration
If your solution incorporates artificial intelligence or machine learning components, the implementation, documentation, and effectiveness of these technologies will be evaluated.

### Business Value & Innovation
Your solution will be assessed on its practical applicability, innovation in addressing the problem statement, and potential real-world impact for the specified use case.


