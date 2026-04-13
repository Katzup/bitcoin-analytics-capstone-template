Can you beat Bitcoin DCA with a simple signal?

For my Georgia Tech OMSA practicum, I tried. The premise was well-motivated: technical analysis has deep roots (Magee & Edwards established the visual pattern framework decades ago), and recent J.P. Morgan research demonstrated that CNNs can classify candlestick chart patterns with high accuracy. So the original hypothesis — train a CNN on candlestick images to predict next-day price direction — seemed reasonable.

It wasn't. On the tournament's actual evaluation metric (recency-weighted sats-per-dollar across 3,076 rolling windows), the CNN scored 41.43%. A neutral baseline that buys the same amount every day? 41.94%.

Classification accuracy ≠ trading edge.

What actually worked was embarrassingly simple: take the OLS trend slope of Bitcoin's log price over a 60-day window, normalize it as a z-score against a 252-day rolling standard deviation, clip outliers, and convert to a daily purchase weight.

Result: 44.95% tournament percentile — a +3.01 percentage point improvement over neutral DCA.

The key insight isn't prediction. It's regime awareness. Bitcoin's volatility clusters. Bull and bear periods have statistically distinct return distributions (Mann-Whitney U test, p < 0.001). By normalizing the trend against its own recent volatility, the signal automatically adapts — spending slightly less during overextended conditions and slightly more during depressed ones.

Several ideas that seemed promising added no value: volatility-amplitude scaling was redundant (the z-score already captures volatility implicitly), weekly resampling underperformed by -0.90 percentage points, and proportional transaction fees are mathematically percentile-invariant in this evaluation framework.

Sometimes the best model is the simplest one that accounts for structure in the data.

Deep dive (Substack): https://bobkatz2.substack.com/p/when-your-cnn-is-a-coin-flip-what
Full analysis, code, and reproducibility package: https://github.com/Katzup/bitcoin-analytics-capstone-template
7-minute video walkthrough: https://youtu.be/dPQTMGjO_cM

This work was completed as part of Georgia Tech's OMSA Practicum in partnership with the Trilemma Foundation (https://www.stackingsats.org/).

#Bitcoin #QuantFinance #DataScience #GeorgiaTech #OMSA #DCA #MachineLearning #Trilemma
