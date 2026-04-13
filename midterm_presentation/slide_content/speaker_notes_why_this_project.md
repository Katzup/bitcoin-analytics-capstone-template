# Speaker Notes: Why Visual Trading Systems?
## Slide 2 — Complete 2-Minute Narrative

---

## The 20-Second Hook (0:00-0:20)

**[Opening — pause, then speak]**

"Before I show you what we built, let me explain why this project matters to me — and why the answer surprised even me."

**[Brief pause to establish connection]**

"This practicum is really a story about disproving your own hypothesis. I started with a belief that visual pattern recognition could beat numerical models. I ended with evidence that sometimes the simplest approach is the most robust."

---

## Part 1: The MBA Foundation (0:20-0:50)

**[Tone: Nostalgic, establishing credibility]**

"My interest in visual trading started back in business school, reading Magee and Edwards' *Technical Analysis of Stock Trends*. This is the classic text — some would say the Bible — of chart pattern analysis."

"What struck me then, and what stayed with me, was the core premise: human traders develop an intuition for visual patterns. They can look at a chart and see a head-and-shoulders formation, or identify support and resistance levels, or recognize trendlines — all without calculating a single indicator."

"The underlying assumption was that price history encodes information about future direction, but it's information that's best accessed *visually*, not numerically. Quants see numbers. Traders see charts."

---

## Part 2: The OMSA Discovery (0:50-1:25)

**[Tone: Excited, building to the research connection]**

"Fast forward to Georgia Tech's OMSA program, and I discovered something fascinating. There's active research coming out of JP Morgan's AI group — and I should note, working with Professor Baich, who taught our ML4T course here at GT — showing that convolutional neural networks could classify candlestick patterns with F1 scores above 0.90."

**[Pause for emphasis]**

"This wasn't theoretical. This was a major investment bank deploying computer vision for trading. They were literally teaching machines to 'see' chart patterns the way human traders do."

"And the results were impressive — on the classification task. But here's the gap those papers didn't fully address: they showed CNNs can *classify* patterns accurately, but does that classification actually lead to *profitable trading decisions* under real-world constraints?"

---

## Part 3: The Tournament Test (1:25-1:50)

**[Tone: Transition to methodology]**

"That's where the Stacking Sats tournament came in. It provided the perfect testbed. Real Bitcoin data from 2016 to 2025. Strict causality enforcement — meaning no lookahead bias, you can only use information you would have had at that moment. Rolling window evaluation across over 3,000 trading days."

"If visual encoding truly provides an edge, it should show up here, under these rigorous conditions. And if it doesn't — well, that's valuable knowledge too. It's just as important to know what *doesn't* work as what does."

---

## Part 4: The Pivot Preview (1:50-2:10)

**[Tone: Setting up the narrative arc]**

"I'll be honest — going into this, I expected the complex CNN to win. I mean, 296,000 parameters, sophisticated architecture, temperature calibration, the whole works. What I found through systematic ablation testing genuinely surprised me."

"It changed how I think about model complexity in finance. And it led us to a simpler, more robust solution that outperformed our sophisticated model on every metric that mattered."

**[Transition to next slide]**

"So let me walk you through what we built, what we tested, and what we learned."

---

## Alternative Shorter Version (90 seconds)

If you need a tighter version:

"My interest in visual trading started with Magee and Edwards' classic technical analysis text — learning how human traders 'see' patterns that quantitative models miss. Years later in OMSA, I discovered research from JP Morgan's AI group with Professor Baich showing CNNs could classify chart patterns with over 90% accuracy."

"But here's the gap: classifying patterns isn't the same as making profitable trades under real constraints. The Stacking Sats tournament gave us the perfect testbed — real BTC data, strict causality, rigorous evaluation."

"I expected our complex CNN to win. What we discovered through systematic ablations surprised us — and changed how I think about model complexity in finance."

---

## Key Phrases to Emphasize

**[Speak slowly and clearly]**
- "Magee and Edwards' *Technical Analysis of Stock Trends*" — book title italicized
- "Professor Baich" — name clearly stated
- "JP Morgan's AI group" — establishes industry credibility
- "296,000 parameters" — emphasize the complexity
- "Systematic ablation testing" — emphasize the rigor

**[Pause after these for effect]**
- "...surprised even me"
- "...valuable knowledge too"
- "...changed how I think about model complexity"

---

## Handling Potential Questions

### Q: "Why mention the book? Isn't that old-school?"
**A:** "Exactly. The contrast between classical technical analysis and modern AI is the point. We're testing whether decades of visual trading intuition can be automated."

### Q: "What was the JP Morgan research specifically?"
**A:** "They were using CNNs to classify candlestick patterns from chart images. High accuracy on the classification task. Our question was whether that translates to allocation decisions."

### Q: "Why this tournament specifically?"
**A:** "The constraints mirror real trading: causality enforcement prevents overfitting, the metric rewards consistency, and the timeframe includes multiple BTC cycles. It's a stress test."

---

## Body Language Tips

- **0:00-0:10:** Stand still, make eye contact, establish presence
- **0:20 (Magee & Edwards):** Open hand gesture — "this is the foundation"
- **0:50 (JP Morgan):** Lean forward slightly — "here's where it gets interesting"
- **1:25 (tournament):** Use both hands to frame "constraints"
- **1:50 (surprise):** Slight head shake — "genuinely surprised me"

---

## Backup Transitions

If you need to cut for time, skip to:
> "My interest started with technical analysis in my MBA, was reinforced by JP Morgan AI research at GT, and the tournament gave us a rigorous test. I expected the CNN to win — but our ablations showed otherwise."

(30 seconds instead of 2 minutes)

---

## Connection to Next Slide

End with:
> "Let me walk you through the tournament constraints and what we actually built."

Then transition smoothly to Slide 3 (The Tournament Challenge).
