# AWS Project Report

Generated 2026-10-01 with read-only CLI checks in us-east-2 (plus one quota request, see below).

## Identity
- Profile: `swirl`
- Account ID: `156084119557` (confirmed by the user as the right project)
- Role: `AccountFullAccessRole`
- Region: us-east-2

## Account plan
- `aws freetier get-account-plan-state`: plan type **PAID**, status **ACTIVE**
- Remaining credits: **120.00 USD**
- `get-free-tier-usage` returned no entries.

## vCPU quotas (us-east-2, applied values)
| Quota | Code | Value |
|---|---|---|
| Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances | L-1216C47A | **16 vCPUs** |
| Running On-Demand F / DL / G and VT / Inf / Trn / P / X / HPC / High Memory | various | 0 |
| All Standard Spot Instance Requests | L-34B43A08 | 32 (moot, spot is denied by policy) |

c8g and c7i are C-family, so they draw on the Standard quota. The other families are 0 and are not relevant to CPU physics runs.

At 16 vCPUs, one c8g.4xlarge or c7i.4xlarge (16 vCPUs) fits. An 8xlarge (32 vCPUs) or 16xlarge (64 vCPUs) does not fit until the increase is granted.

## Instance offerings and on-demand prices (us-east-2, Linux, shared tenancy)
All six sizes are offered in us-east-2.

| Type | vCPUs | USD/hour |
|---|---|---|
| c8g.4xlarge | 16 | 0.6358 |
| c8g.8xlarge | 32 | 1.2717 |
| c8g.16xlarge | 64 | 2.5434 |
| c7i.4xlarge | 16 | 0.7140 |
| c7i.8xlarge | 32 | 1.4280 |
| c7i.16xlarge | 64 | 2.8560 |

Source: AWS Price List API. c8g is Graviton (arm64) and c7i is Intel (x86_64). The 120 USD credit covers about 47 hours of one c8g.16xlarge or about 42 hours of one c7i.16xlarge.

## Spot is denied by policy
From https://docs.aws.amazon.com/accounts/latest/reference/scps-and-rcps-for-projects.html, the service control policies AWS applies to projects include explicit denies:
- `DenyRequestSpotInstances`: Deny `ec2:RequestSpotInstances` on `*`
- `DenySpotMarketType`: Deny `ec2:RunInstances` where `ec2:InstanceMarketType` = `spot`

Spot requests are therefore refused, and **only on-demand capacity can be used**. No spot request was attempted. The same policy also denies Reserved Instance and Savings Plan purchases.

## Quota increase request
- Quota: Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances, us-east-2 (L-1216C47A)
- Requested: 16 -> **64 vCPUs**
- Request ID: `f00511d0f93147afb7d6b704be5dec67pQ8AD9jr`
- Status: **PENDING** (as of 2026-10-01, filed 13:53 +07:00)

To check on it later:
```
aws service-quotas get-requested-service-quota-change --request-id f00511d0f93147afb7d6b704be5dec67pQ8AD9jr --profile swirl --region us-east-2
```
