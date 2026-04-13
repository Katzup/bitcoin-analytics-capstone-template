"""
Generate a simple HTML presentation that can be opened in a browser
Run: python generate_html_slides.py
Then open: presentation.html
"""

import os

OUTPUT_FILE = "/Users/bobkatz/Visual_Trading_System/midterm_presentation/presentation.html"
IMAGES_DIR = "images"

def generate_html():
    html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VTS Midterm Presentation</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a2e;
            color: #fff;
            overflow: hidden;
        }
        
        .slide {
            width: 100vw;
            height: 100vh;
            display: none;
            padding: 60px 80px;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            position: relative;
        }
        
        .slide.active {
            display: flex;
        }
        
        .slide h1 {
            font-size: 3.5em;
            margin-bottom: 30px;
            color: #fff;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .slide h2 {
            font-size: 2.5em;
            margin-bottom: 40px;
            color: #4ecca3;
        }
        
        .slide h3 {
            font-size: 1.8em;
            margin: 20px 0;
            color: #ff6b6b;
        }
        
        .slide p, .slide li {
            font-size: 1.4em;
            line-height: 1.6;
            max-width: 900px;
            margin: 15px 0;
        }
        
        .slide ul {
            text-align: left;
            list-style-position: inside;
        }
        
        .slide img {
            max-width: 85%;
            max-height: 60vh;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            margin: 20px 0;
        }
        
        .big-number {
            font-size: 4em;
            font-weight: bold;
            color: #4ecca3;
            margin: 20px 0;
        }
        
        .highlight {
            color: #ff6b6b;
            font-weight: bold;
        }
        
        .highlight-green {
            color: #4ecca3;
            font-weight: bold;
        }
        
        .navigation {
            position: fixed;
            bottom: 30px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 15px;
            z-index: 1000;
        }
        
        .nav-btn {
            padding: 12px 24px;
            background: #4ecca3;
            color: #1a1a2e;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .nav-btn:hover {
            background: #3db892;
            transform: scale(1.05);
        }
        
        .slide-counter {
            position: fixed;
            bottom: 30px;
            right: 40px;
            font-size: 1.2em;
            color: #888;
        }
        
        .two-column {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 40px;
            width: 100%;
            max-width: 1200px;
        }
        
        .metric-box {
            background: rgba(255,255,255,0.1);
            padding: 30px;
            border-radius: 15px;
            margin: 10px;
        }
        
        .metric-box h3 {
            color: #fff;
            margin-bottom: 15px;
        }
        
        .quote {
            font-style: italic;
            font-size: 1.6em;
            border-left: 5px solid #4ecca3;
            padding-left: 30px;
            margin: 30px 0;
            text-align: left;
        }
        
        table {
            width: 80%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 1.2em;
        }
        
        th, td {
            padding: 15px;
            border: 1px solid #444;
            text-align: center;
        }
        
        th {
            background: #4ecca3;
            color: #1a1a2e;
        }
        
        tr:nth-child(even) {
            background: rgba(255,255,255,0.05);
        }
        
        .title-slide {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        }
        
        .title-slide h1 {
            font-size: 4em;
            margin-bottom: 20px;
        }
        
        .title-slide p {
            font-size: 1.8em;
            color: #888;
        }
        
        .emoji {
            font-size: 1.5em;
        }
    </style>
</head>
<body>
'''

    slides = [
        # Slide 1: Title
        '''
    <div class="slide active title-slide">
        <h1>When AI Meets Market Reality</h1>
        <p style="font-size: 2em; color: #ff6b6b; margin: 30px 0;">A 296K-Parameter CNN That Lost to a Coin Flip</p>
        <p style="margin-top: 60px;">VTS Tournament Submission - Stacking Sats Challenge</p>
        <p style="color: #666; margin-top: 20px;">Practicum Midterm Presentation</p>
    </div>
''',
        
        # Slide 2: The Challenge
        '''
    <div class="slide">
        <h2>The Tournament Challenge</h2>
        <div style="text-align: left; max-width: 800px;">
            <p><span class="emoji">🎯</span> <strong>Objective:</strong> Predict daily BTC allocation weights (2016-2025)</p>
            <p><span class="emoji">📊</span> <strong>Metric:</strong> Recency-Weighted SPD Percentile</p>
            <p><span class="emoji">⚖️</span> <strong>Constraints:</strong> Σwᵢ = 1.0, wᵢ ≥ 1e-5</p>
            <p><span class="emoji">🔒</span> <strong>Critical:</strong> Strict causality - no lookahead allowed</p>
            <p><span class="emoji">🏆</span> <strong>Target:</strong> Top performers >60% percentile</p>
        </div>
    </div>
''',
        
        # Slide 3: Our Approach
        '''
    <div class="slide">
        <h2>Our Initial Approach</h2>
        <div style="font-family: monospace; text-align: left; background: rgba(0,0,0,0.3); padding: 30px; border-radius: 10px;">
            <p>Price Data (90 days)</p>
            <p style="color: #888;">↓</p>
            <p><strong>GAF Image Generation</strong></p>
            <p style="color: #888;">↓</p>
            <p><strong>Deep CNN (296K params)</strong></p>
            <p style="color: #888;">↓</p>
            <p>Temperature Calibration → Tilt Allocation → EMA → Normalize</p>
        </div>
        <p style="margin-top: 30px; color: #4ecca3;">Hypothesis: Visual patterns in GAF images predict returns</p>
    </div>
''',
        
        # Slide 4: First Results
        '''
    <div class="slide">
        <h2>First Results: Reality Check</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>CNN Result</th>
                <th>Target</th>
            </tr>
            <tr>
                <td>RW SPD Percentile</td>
                <td class="highlight">41.43%</td>
                <td class="highlight-green">>60%</td>
            </tr>
            <tr>
                <td>Win Rate vs DCA</td>
                <td>54.32%</td>
                <td class="highlight-green">>50%</td>
            </tr>
            <tr>
                <td>Windows Evaluated</td>
                <td colspan="2">3,076 rolling windows</td>
            </tr>
        </table>
        <p style="margin-top: 40px; font-size: 1.8em;">
            <span class="highlight">Underperforms</span> simple Dollar-Cost Averaging
        </p>
    </div>
''',
        
        # Slide 5: Ablation 1
        '''
    <div class="slide">
        <h2>Ablation Study #1: Allocator Sizing</h2>
        <p style="margin-bottom: 30px;"><strong>Question:</strong> Are losses from overbetting?</p>
        <img src="images/01_ablation_sensitivity.png" alt="Sensitivity Sweep">
        <p style="color: #ff6b6b; margin-top: 20px;">
            <span class="emoji">⚠️</span> Conclusion: Minimal impact (+0.35 pp max). Problem NOT allocator aggressiveness.
        </p>
    </div>
''',
        
        # Slide 6: Ablation 2 (KEY SLIDE)
        '''
    <div class="slide">
        <h2>Ablation Study #2: The "Aha!" Moment</h2>
        <p style="margin-bottom: 20px;"><strong>Test:</strong> Replace CNN with constant prob_up = 0.5 (coin flip)</p>
        <img src="images/02_ablation_signal_quality.png" alt="Signal Quality Test">
        <p style="color: #4ecca3; font-size: 1.6em; margin-top: 20px;">
            <span class="emoji">✅</span> CNN signal is WORTHLESS. Coin flip outperforms with higher consistency.
        </p>
    </div>
''',
        
        # Slide 7: Ablation 3
        '''
    <div class="slide">
        <h2>Ablation Study #3: EMA Smoothing</h2>
        <p style="margin-bottom: 20px;"><strong>Question:</strong> Can smoothing fix the performance?</p>
        <img src="images/03_ablation_ema_flat.png" alt="EMA Flat Line">
        <p style="color: #ff6b6b; margin-top: 20px;">
            <span class="emoji">📉</span> Zero effect across ALL alphas. Mathematical proof: signal quality is the issue.
        </p>
    </div>
''',
        
        # Slide 8: Root Causes
        '''
    <div class="slide">
        <h2>Why GAF + CNN Failed</h2>
        <img src="images/06_root_causes.png" alt="Root Causes">
    </div>
''',
        
        # Slide 9: The Pivot
        '''
    <div class="slide">
        <h2>The Evidence-Based Pivot</h2>
        <div style="text-align: left; max-width: 800px;">
            <h3 style="color: #4ecca3;">Simplified Baseline</h3>
            <p>• Constant prob_up = 0.5 (neutral probability)</p>
            <p>• Same allocation logic, NO model artifacts</p>
            <p>• 10x faster, more robust, higher win rate</p>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 40px;">
                <div class="metric-box">
                    <h3>CNN Approach</h3>
                    <p>41.43% RW</p>
                    <p>54.32% Win Rate</p>
                    <p>1.1 MB Model</p>
                </div>
                <div class="metric-box" style="border: 2px solid #4ecca3;">
                    <h3 style="color: #4ecca3;">Simplified ⭐</h3>
                    <p class="highlight-green">41.94% RW</p>
                    <p class="highlight-green">70.42% Win Rate</p>
                    <p class="highlight-green">No Model</p>
                </div>
            </div>
        </div>
    </div>
''',
        
        # Slide 10: Performance Comparison
        '''
    <div class="slide">
        <h2>Side-by-Side Comparison</h2>
        <img src="images/04_performance_comparison.png" alt="Performance Comparison">
        <p style="color: #4ecca3; font-size: 1.6em; margin-top: 20px;">
            <span class="emoji">🏆</span> Simplified wins on ALL dimensions
        </p>
    </div>
''',
        
        # Slide 11: Engineering Wins
        '''
    <div class="slide">
        <h2>What We Did Right</h2>
        <div class="two-column">
            <div style="text-align: left;">
                <h3 style="color: #4ecca3;">Technical Excellence</h3>
                <p><span class="emoji">✓</span> Strict causality enforcement</p>
                <p><span class="emoji">✓</span> Deterministic (seed=42)</p>
                <p><span class="emoji">✓</span> Comprehensive testing</p>
                <p><span class="emoji">✓</span> Grader-safe implementation</p>
            </div>
            <div style="text-align: left;">
                <h3 style="color: #4ecca3;">Scientific Rigor</h3>
                <p><span class="emoji">✓</span> Hypothesis-driven development</p>
                <p><span class="emoji">✓</span> Systematic ablation studies</p>
                <p><span class="emoji">✓</span> Transparent negative results</p>
                <p><span class="emoji">✓</span> Learning orientation</p>
            </div>
        </div>
    </div>
''',
        
        # Slide 12: Key Lessons
        '''
    <div class="slide">
        <h2>Key Lessons Learned</h2>
        <div style="text-align: left; max-width: 900px;">
            <p style="margin: 25px 0;"><span class="highlight">1.</span> <strong>Ablation studies are critical</strong> - Without them, we'd never know CNN was worthless</p>
            <p style="margin: 25px 0;"><span class="highlight">2.</span> <strong>Test "better than random?" early</strong> - Should be Ablation #0, not #2</p>
            <p style="margin: 25px 0;"><span class="highlight">3.</span> <strong>Complexity ≠ Performance</strong> - 296K parameters lost to coin flip</p>
            <p style="margin: 25px 0;"><span class="highlight">4.</span> <strong>Simple often beats complex</strong> - Occam's Razor applies to trading</p>
        </div>
    </div>
''',
        
        # Slide 13: Future Work
        '''
    <div class="slide">
        <h2>Future Improvements</h2>
        <img src="images/07_future_improvements.png" alt="Future Improvements">
        <p style="margin-top: 20px;">
            <strong>Highest ROI:</strong> Weekly granularity + Regime detection
        </p>
        <p style="color: #4ecca3;">
            Expected: 41.94% → 57-73% RW percentile
        </p>
    </div>
''',
        
        # Slide 14: Conclusion
        '''
    <div class="slide">
        <h2>The Bottom Line</h2>
        <div class="quote">
            "We built a technically sophisticated system and discovered through rigorous testing that it performs equivalently to a coin flip. This simplified version represents our evidence-based conclusion: in financial time series, signal quality matters more than model complexity."
        </div>
        <p style="margin-top: 40px; font-size: 1.8em; color: #4ecca3;">
            <span class="emoji">🎓</span> Scientific Integrity > Raw Performance
        </p>
    </div>
''',
        
        # Slide 15: Thank You
        '''
    <div class="slide title-slide">
        <h1>Thank You</h1>
        <p style="font-size: 1.8em; margin: 30px 0;">Questions?</p>
        <div style="margin-top: 60px; text-align: left; max-width: 600px;">
            <p style="font-size: 1em; color: #666; margin: 10px 0;">
                <strong>Documentation:</strong> LESSONS_LEARNED.md
            </p>
            <p style="font-size: 1em; color: #666; margin: 10px 0;">
                <strong>Code:</strong> Visual_Trading_System repository
            </p>
            <p style="font-size: 1em; color: #666; margin: 10px 0;">
                <strong>Notebooks:</strong> btc_accumulation_model_simplified.ipynb
            </p>
        </div>
        <p style="margin-top: 60px; color: #4ecca3; font-style: italic;">
            "The best model is the one that works, not the one that sounds impressive."
        </p>
    </div>
'''
    ]
    
    # Add all slides
    html += '\n'.join(slides)
    
    # Add navigation
    html += '''
    <div class="navigation">
        <button class="nav-btn" onclick="prevSlide()">← Previous</button>
        <button class="nav-btn" onclick="nextSlide()">Next →</button>
    </div>
    
    <div class="slide-counter">
        <span id="current">1</span> / <span id="total">''' + str(len(slides)) + '''</span>
    </div>
    
    <script>
        let currentSlide = 0;
        const slides = document.querySelectorAll('.slide');
        const totalSlides = slides.length;
        
        document.getElementById('total').textContent = totalSlides;
        
        function showSlide(n) {
            slides.forEach(slide => slide.classList.remove('active'));
            slides[n].classList.add('active');
            document.getElementById('current').textContent = n + 1;
        }
        
        function nextSlide() {
            currentSlide = (currentSlide + 1) % totalSlides;
            showSlide(currentSlide);
        }
        
        function prevSlide() {
            currentSlide = (currentSlide - 1 + totalSlides) % totalSlides;
            showSlide(currentSlide);
        }
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowRight' || e.key === ' ') nextSlide();
            if (e.key === 'ArrowLeft') prevSlide();
        });
    </script>
</body>
</html>
'''
    
    with open(OUTPUT_FILE, 'w') as f:
        f.write(html)
    
    print(f"✅ HTML presentation generated: {OUTPUT_FILE}")
    print(f"📊 Total slides: {len(slides)}")
    print("\nTo view:")
    print("  1. Open presentation.html in a web browser")
    print("  2. Use arrow keys or buttons to navigate")
    print("  3. Press F11 for fullscreen presentation mode")

if __name__ == "__main__":
    generate_html()
