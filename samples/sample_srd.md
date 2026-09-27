# Customer Loyalty Platform – Software Requirements Specification

## 1. Architecture
SR-01: The platform is a single Java application deployed on one virtual machine with a PostgreSQL database.
SR-02: The system shall support 5,000 concurrent users.

## 2. Points processing
SR-03: Purchases are collected from store systems in a nightly batch file and points are credited the next morning.
SR-04: Point redemption calls the payment gateway synchronously with no timeout or retry defined.

## 3. Data
SR-05: Member balances are exported from the AS400 system as CSV and loaded once at go-live.

## 4. Availability
SR-06: The system shall be available 99.5% during business hours.

## 5. Security
SR-07: Users log in with email and password.
