# AWS setup for rented compute

Date: 2026-10-01 (revision 2, after reading AWS's documentation for the new experience and Aaron's AWS Settings screen)

## Where things stand

| Item | Status |
|---|---|
| AWS account, new experience | Done |
| Paid plan | Done |
| Credits | 100 dollars available |
| Default project "Hello World" | Exists, with a 20 dollar spend limit. Not used for this work |
| Dedicated project for SWIRL | To create (Part A) |
| Mac connected to the project | To do (Part B) |
| Core quota raised | To do (Part B requests it; AWS approves it) |

## Facts that shape the setup

- **Region is fixed.** Every project in this account is in US East (Ohio), us-east-2. AWS assigns the Region from the contact address at sign-up, and a new project gets the same one. Distance does not matter for this work: the simulations run on the rented machine, and only commands and small result files travel.
- **A project has one setting at creation: its name.**
- **Spend limits are per project.** A new project has no limit until you set one.
- **Spend limits exclude credits.** The limit caps real charges only. The 50 dollar working budget is enforced by the harness, through the cost ledger and automatic shutdown.
- **Spot machines are blocked** in the new experience, on both plans. We pay the on-demand price.
- **New accounts start with a small core quota.** The increase request is the slow step, so it goes in early.

## Part A. Create the project (you, in the browser, about 2 minutes)

1. Open https://settings.aws.com and choose **Projects**.
2. Choose **Create project**. Enter a name and choose **Create**.
3. Choose **Billing**. Under **Cost by project**, find the new project and choose **Set limit**. Choose a custom limit of 20 dollars and choose **Set spend limit**.
4. Back on **Projects**, note the new project's account ID. Part B checks against it.

Leave "Hello World" as it is. Nothing runs in it and it costs nothing. Leave "Activate advanced features" alone as well.

## Part B. Connect the Mac and request the quota (a Code session, about 15 minutes)

Do this after unzipping the SWIRL kit, because the session saves its report into `~/Desktop/SWIRL/docs/`.

1. Create a new empty folder on your Mac for this. Any name works. It keeps AWS's own rules file out of the SWIRL sessions.
2. Open a new Code session on that folder, on Sonnet.
3. Paste the prompt below.
4. Code asks for a profile name. Use the project's name, so the two are easy to match.
5. A browser window opens. Sign in with the login you used for AWS. On the **Choose AWS sessions** page, pick the new project, not Hello World.
6. Code runs a setup wizard. Answer its prompts; Code says what each one is for.
7. Code shows you the account ID it is connected to. Check it matches the new project before it goes on.
8. Code saves `AWS_REPORT.md` into `~/Desktop/SWIRL/docs/` and files the quota request.

Your sign-in lasts 12 hours and renews for up to 90 days. When it lapses, Code runs `aws login` and you sign in again in the browser.

```
Objective: Connect this Mac to my AWS project by following the instructions at
https://raw.githubusercontent.com/aws/agent-toolkit-for-aws/refs/heads/main/setup-instructions/setup.md
then report what the project allows and request a core quota increase.

Inputs:
- AWS Region: us-east-2
- AWS experience: the new AWS experience
- Profile name: ask me before starting, as the instructions require.

Intent: A separate project folder (~/Desktop/SWIRL) will rent compute in this AWS project for physics simulations. This session makes the connection and gets the quota request moving. It launches nothing.

Setup: Follow the setup file's steps in order. I complete the browser sign-in (Step 3) and the interactive wizard (Step 5) myself; pause for each. Step 7 writes its rules file into this folder, which is intended.

Check the project: After sign-in, show me the account ID from aws sts get-caller-identity and wait for me to confirm it is the right project.

Report: Run these read-only checks in us-east-2 with the same profile and save the results as ~/Desktop/SWIRL/docs/AWS_REPORT.md (if that folder does not exist, save it here and tell me):
- Profile name and account ID.
- Account plan type (aws freetier get-account-plan-state) and credit balance if the CLI can show it.
- Current vCPU quota for Running On-Demand Standard instances, and the applied quota for any other EC2 instance family that looks relevant.
- Whether c8g.16xlarge, c7i.16xlarge and their 8xlarge and 4xlarge sizes are offered in us-east-2, with current on-demand prices.
- Confirmation that a spot request is denied by policy, from the documentation at https://docs.aws.amazon.com/accounts/latest/reference/scps-and-rcps-for-projects.html, without attempting one.

Quota request: Once I have confirmed the project, request an increase of the Running On-Demand Standard instances quota in us-east-2 to 64 vCPUs through Service Quotas. Add the request ID and its status to the report.

Constraints: Apart from that one quota request, do not create, launch or modify any AWS resource. Do not change the plan, billing, spend limits or advanced features. Never ask me for access keys or secret keys.

Done when: aws sts get-caller-identity succeeds under the chosen profile for the project I confirmed, aws agent-toolkit list-available-skills returns the skill list, the aws-mcp entry carries the profile as the instructions describe, the report file exists, and the quota request is filed or its refusal is recorded with the reason.
```

## How it connects to SWIRL

The AWS CLI and your sign-in live on the Mac, not in a folder, so the SWIRL session uses them through the profile name in `AWS_REPORT.md`. The `cloud` worker in the SWIRL harness launches a machine in us-east-2, runs the batch, copies results back, shuts the machine down, and records the cost in `docs/CLOUD_LEDGER.md`. Until the quota is approved, SWIRL runs on the Mac.

## Not included yet

- **An AWS-side alert at 50 dollars.** It needs an email address to notify. Tell Claude Desktop which address and it gets added.
