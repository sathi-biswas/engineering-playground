# Sample Bug Report — BUG-1042

**Title:** `calculate_discount` returns incorrect value for VIP customers

## Description

When a customer with `tier="vip"` checks out, the discount applied is 5% instead of the documented 15%.

## Error / Observed Behavior

```
AssertionError: expected discount 15.0, got 5.0
  File "shop/pricing.py", line 42, in calculate_discount
    return base_rate * subtotal
```

## Stack Trace

```
Traceback (most recent call last):
  File "tests/test_pricing.py", line 18, in test_vip_discount
    assert calculate_discount(100, tier="vip") == 15.0
AssertionError
```

## Suspected Components

- `shop/pricing.py` — `calculate_discount`
- Possibly `shop/tiers.py` rate table

## Reproduction Steps

1. Call `calculate_discount(subtotal=100, tier="vip")`
2. Observe return value `5.0` instead of `15.0`
