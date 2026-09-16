# GloCare Intent Annotation Guidelines

## General rule

Assign exactly ONE primary intent to each customer message.

Choose the intent that best represents the customer's primary support need or
the support action required.

Do not assign an intent only because a keyword appears in the message.

If the message does not contain enough actionable information to determine a
meaningful support intent, use `non_actionable_or_context_required`.

---

## 1. network_issue

### Meaning

Problems involving Glo network availability, connectivity, coverage, signal,
or general network performance.

### Include when

- Network is unavailable or disappears.
- Customer reports poor/weak/fluctuating network.
- Customer cannot connect to the network.
- Customer asks whether Glo network/4G is available in an area.
- Internet/calling problems are described primarily as a network problem.

### Do NOT use when

- The primary problem is an unauthorized deduction or VAS.
- The primary problem is recharge/airtime.
- The primary problem is SIM replacement or SIM swap.
- The primary problem is a specific data-plan subscription issue.

### Boundary

If the customer explicitly describes a network/connectivity problem affecting
multiple services, prefer `network_issue`.

---

## 2. data_issue

### Meaning

Problems with an existing data service, data balance, data allocation,
data usage, or ability to browse/use data.

### Include when

- Customer has data but cannot browse.
- Data was deducted unexpectedly.
- Customer received less data than expected.
- Customer says data finished unexpectedly.
- Existing data service is not working.

### Do NOT use when

- Customer is primarily asking about buying/subscribing/renewing a data plan.
- Customer is primarily asking about a promotion or bonus.

### Boundary

Existing data/service problem → `data_issue`.

Data-plan purchase, subscription, renewal, or plan information → `data_plan_support`.

---

## 3. data_plan_support

### Meaning

Questions or problems related to data plans, bundles, subscription,
renewal, pricing, eligibility, or plan features.

### Include when

- Customer asks about a data plan.
- Customer wants to subscribe to a data plan.
- Customer wants to renew or stop auto-renewal.
- Customer asks about plan price, size, validity, or features.
- Customer asks about a data bundle/hotspot offering.

### Do NOT use when

- The customer already has data and it is not working.
- The primary problem is an unauthorized deduction.

### Boundary

Plan/subscription information or action → `data_plan_support`.

Existing data service failure/usage problem → `data_issue`.

---

## 4. unauthorized_charge_or_vas

### Meaning

Unexpected, unauthorized, unwanted, or disputed airtime/account deductions,
including Value Added Services (VAS) or subscriptions.

### Include when

- Customer reports money/airtime deducted unexpectedly.
- Customer says they did not subscribe to a service.
- Customer wants to unsubscribe from VAS.
- Customer requests a refund for an unauthorized deduction.

### Do NOT use when

- The deduction is clearly part of an expected data-plan purchase and there is
no complaint about unauthorized charging.
- The primary issue is simply failed recharge.

### Boundary

Unexpected deduction / unwanted service / VAS → `unauthorized_charge_or_vas`.

---

## 5. recharge_or_airtime_issue

### Meaning

Problems with recharging the Glo line or loading/receiving airtime credit.

### Include when

- Recharge PIN/card does not work.
- Recharge fails.
- Customer cannot load airtime.
- Customer asks about recharge status.
- Customer reports an issue specifically with adding airtime.

### Do NOT use when

- Airtime was deducted unexpectedly → `unauthorized_charge_or_vas`.
- The issue is primarily about a data plan.

### Boundary

Adding/recharging airtime → `recharge_or_airtime_issue`.

Unexpected loss/deduction of airtime → `unauthorized_charge_or_vas`.

---

## 6. sim_or_device_support

### Meaning

Support involving the physical SIM, SIM replacement/swap, SIM credentials,
or device compatibility/setup.

### Include when

- SIM is lost.
- Customer needs SIM replacement.
- Customer needs SIM swap.
- Customer asks about SIM compatibility.
- Customer provides/asks about SIM serial or PUK information.
- Customer asks whether a device supports a Glo service.

### Do NOT use when

- The primary complaint is general network availability.
- The primary complaint is simply inability to browse with an otherwise
  functioning SIM/service.

### Boundary

SIM/device-specific support → `sim_or_device_support`.

General connectivity problem → `network_issue`.

---

## 7. voice_or_line_support

### Meaning

Problems primarily involving voice calls, calling capability, or the
customer's line/service for voice communication.

### Include when

- Customer cannot make calls.
- Customer cannot receive calls.
- Customer reports a calling/voice-specific problem.
- Customer reports a line-specific voice service problem.

### Do NOT use when

- The complaint is primarily about network availability/coverage.
- The primary problem is recharge or unauthorized deduction.

### Boundary

If the customer describes a broad network/connectivity failure affecting
multiple services, prefer `network_issue`.

If the problem is specifically about making/receiving calls or voice service,
use `voice_or_line_support`.

---

## 8. bonus_or_promotion_issue

### Meaning

Problems or questions involving Glo bonuses, promotional offers, bonus
eligibility, bonus balances, or promotional benefits.

### Include when

- Customer did not receive an expected bonus.
- Customer asks how to check a bonus.
- Customer asks about bonus eligibility.
- Customer asks about promotion benefits or validity.
- Customer reports a problem with promotional data/airtime.

### Do NOT use when

- The issue is a normal data-plan problem without a bonus/promotion component.
- The issue is an unauthorized VAS deduction.

---

## 9. account_or_general_support

### Meaning

Account, line, or general Glo support requests that do not fit a more
specific support category.

### Include when

- Customer needs account/line-specific assistance that does not fit another
  intent.
- Customer asks a general Glo support question that does not belong to a
  specific service category.
- Customer asks how to access or manage a general account/line function.

### Do NOT use when

A more specific intent clearly applies.

For example:

- Lost SIM → `sim_or_device_support`
- Data plan question → `data_plan_support`
- Unauthorized deduction → `unauthorized_charge_or_vas`
- Network problem → `network_issue`

### Boundary

Use this as a general-support category, not as a catch-all for messages that
clearly belong to another intent.

---

## 10. non_actionable_or_context_required

### Meaning

The message does not contain enough actionable information to determine a
specific support intent.

### Include when

- Customer only says "Thanks", "Okay", "Hello", etc.
- Customer provides only a phone number or other context.
- Customer says they are still waiting without explaining the underlying issue.
- Message depends on previous conversation context that is unavailable.
- Message is too incomplete to identify the support need.

### Do NOT use when

A clear support intent can be identified even if the message is short.

### Boundary

Short does NOT automatically mean non-actionable.

Use this category only when the available message genuinely lacks enough
information to identify the primary support need.

If the customer text itself does not provide enough information to identify
the primary support intent, use this category even if the original
conversation may have contained additional context.

### Network vs data boundary

If the complaint is about Glo's network/connectivity itself
(signal, coverage, network availability, or network repeatedly disconnecting),
use `network_issue`.

If the customer indicates that their existing data allowance/service is being
consumed, unavailable, insufficient, or not functioning despite the data
service being available, use `data_issue`.

Do not classify based only on words such as "internet", "network", or "data".
Use the customer's actual support need.