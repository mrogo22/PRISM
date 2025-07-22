#  Data Requirements for PRISM Framework

>  **Note**: This framework assumes data preparation has already been completed.
> 
> The following input datasets are required to run the core stages of the PRISM pipeline:
> - Feature Engineering
> - Subgroup Discovery using EMM
> - Price Elasticity Forecasting using ARIMA
> - Price Recommendation
>
> These inputs reflect the format used in the original industrial implementation. Users can replace them with equivalent datasets that include:
> - Product-level transactional metrics
> - Price and cost history
> - Inflation adjustments

---

##  Dataset 1: `DataPreparationOutput.csv`

**Description**:  
Cleaned and enriched order-level transactional data. This file contains both raw input and engineered attributes merged from various sources, including order metadata, pricing, cost, OEM information, and supplier lead times.

###  Key Columns

| Column Name                 | Type         | Description                                                            |
|-----------------------------|--------------|------------------------------------------------------------------------|
| `CustomerSoldTo`           | string        | ID of the customer entity                                              |
| `ShipTo`                   | string        | Shipping location or branch                                            |
| `OrderType`                | string        | Type of sales order                                                    |
| `OrderNumber`              | int           | Unique identifier for the sales order                                  |
| `ItemNumber`               | string        | Line item number within the order                                      |
| `Description`              | string        | Product name or summary                                                |
| `PrimaryUoM`               | string        | Unit of measure used for quantity                                      |
| `OrderQuantity`            | float         | Quantity ordered                                                       |
| `Discount`                 | string/object | Discount granted on the order line                                     |
| `BaseCurrency`             | string        | Currency of the transaction                                            |
| `PricingUOM`               | string        | Unit of measure used for pricing                                       |
| `OrderDate_x`              | datetime      | Date when the order was placed                                         |
| `SupplierNumber`           | int           | ID of the supplier                                                     |
| `GeographicRegion`         | string        | Customer region (e.g., 'EU', 'USA', 'APA')                             |
| `OrderYear`, `OrderMonth`, `OrderDay` | int | Parsed date components                                                 |
| `TotalCostEUR`             | float         | Total cost (converted to EUR)                                          |
| `PriceSalesUoMEUR`         | float         | Final sales price per UoM (EUR)                                        |
| `TotalPriceEUR`            | float         | Final total sales price in EUR                                         |
| `CostSalesUoMEUR`          | float         | Internal cost per UoM (EUR)                                            |
| `GLPSalesUoM`              | float         | General List Price per UoM (if available)                              |
| `EffectiveFrom`, `EffectiveThru` | datetime | Price validity interval                                                |
| `PriceCategory`            | string        | Assigned pricing cluster or bucket                                     |
| `SupplierLeadtime`         | float         | Average supplier delivery lead time (in days)                          |
| `PCoverPriceSalesUoM`      | float         | Contract-based price per UoM                                           |
| `TechnicalClassification`  | string        | Engineering category (e.g., 'Mechanical – Category A')                 |
| `VIEngineered`             | float         | Binary flag for custom-engineered products                             |
| `SparePartsCategory`       | string        | Category label (e.g., 'Durable', 'Wear')                               |
| `OEMName`, `OEMItemNumber`| string        | Original manufacturer and OEM-specific product code                    |
| `ActualShipDate`           | datetime      | Date when the product was actually shipped                             |
| `OriginalPromisedShipDate`, `LatestPromisedShipDate` | datetime | Promised delivery dates for the order line                 |

**Note**: Columns with suffixes `_x` and `_y` come from joins with other sources (e.g., pricing systems). Missing values are expected in some OEM and price-related fields.

---

##  Dataset 2: `InflationUpTo2025.csv`

**Description**:  
Official monthly inflation data from the European Central Bank, used to adjust nominal prices to real values (e.g., inflation-adjusted GLP).

###  Columns

| Column Name                                  | Type     | Description                                                        |
|----------------------------------------------|----------|--------------------------------------------------------------------|
| `DATE`                                       | datetime | Start date of the inflation measurement period                     |
| `TIME PERIOD`                                | string   | Human-readable label for the time period (e.g., '2022-03')         |
| `HICP - Overall index (ICP.M.U2.N.000000.4.ANR)` | float    | Harmonized Index of Consumer Prices (HICP) for the euro area        |

**Usage Note**:  
Inflation values are used to compute adjusted price change ratios for elasticity calculation. Matching is typically done on order month/year or GLP active period.

---

##  Assumptions for Users

If you are not using these exact datasets:

- You must provide **a transactional dataset** where:
  - Product-level sales quantities and prices are timestamped
  - Each product has been sold under **at least two different price levels**
- You must provide an **inflation index** or disable inflation adjustment logic

See the notebook `Step 2 Aggregate Price Elasticity.ipynb` for examples on feature aggregation and price elasticity computations. 
