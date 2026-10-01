# Cloud ledger

Every rented-compute dollar is recorded here. Working budget: 50 USD in total (pre-approved). At 40 USD spent, tell Aaron in the report; at 50, stop renting.

Account 156084119557, profile `swirl`, Region us-east-2, on-demand only. Prices from the AWS Price List API.

| Date | Instance type | Hours | Cost (USD) | What ran | Confirmation nothing remains |
|---|---|---|---|---|---|
| 2026-10-01 | c7i.4xlarge (i-0ca8cfde9b83d0c26), 0.714 USD/h | 3.90 (launch 08:24:46Z, terminated about 12:18:50Z) | 2.80 (compute 2.78 + 40 GB gp3 about 0.02; S3 under 0.01) | WP3 Session 1 at commit 3056234: engine tests on the machine (29 passed, 1 xfailed), then run3_primary_quad (N=192, T=1000), run4_primary_res (N=288, T=300), box_L140_N256, box_L70_N128 | 2026-10-01 12:20Z: instance terminated; bucket, objects, IAM role and instance profile, security group deleted; no running instances, volumes, snapshots, addresses, key pairs or buckets left in us-east-2 |

**Ledger total: 2.80 USD** (of the 50 USD budget).
