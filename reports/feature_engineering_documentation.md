# Feature Engineering Documentation

## Engineered Features

### Time-Based Features
- hour_of_day: Hour of transaction
- day_of_week: Day of week
- is_weekend: Weekend indicator
- is_late_night: Late night indicator

### Velocity Features
- transactions_last_1h: Transaction count in past hour
- transactions_last_24h: Transaction count in past day

### User Behavior
- time_since_signup_hours: Time since account creation
- amount_deviation_ratio: Deviation from user average

### IP-to-Country Mapping
- Converted IP addresses to integers
- Performed range-based lookup
- Added country and country_risk features
