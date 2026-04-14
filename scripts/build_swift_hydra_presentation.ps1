param(
    [string]$TemplatePath = "C:\Users\haryd\Downloads\Eric Schwarz Capstone Presentation (1).pptx",
    [string]$OutputPath = "C:\Users\haryd\fed_learn\outputs\dat_le_swift_hydra_presentation.pptx"
)

$ErrorActionPreference = "Stop"

function Get-TitleShape {
    param([object]$Slide)

    foreach ($shape in $Slide.Shapes) {
        try {
            $placeholderType = $shape.PlaceholderFormat.Type
            if ($placeholderType -eq 1 -or $placeholderType -eq 3) {
                return $shape
            }
        } catch {
        }
    }

    throw "No title placeholder found on slide $($Slide.SlideIndex)."
}

function Set-NotesText {
    param(
        [object]$Slide,
        [string]$Text
    )

    foreach ($shape in $Slide.NotesPage.Shapes) {
        try {
            if ($shape.PlaceholderFormat.Type -eq 2) {
                $shape.TextFrame.TextRange.Text = $Text
                return
            }
        } catch {
        }
    }

    throw "No notes placeholder found on slide $($Slide.SlideIndex)."
}

function Reset-ContentSlide {
    param([object]$Slide)

    for ($index = $Slide.Shapes.Count; $index -ge 1; $index--) {
        $shape = $Slide.Shapes.Item($index)
        $keepShape = $false

        try {
            $placeholderType = $shape.PlaceholderFormat.Type
            if ($placeholderType -eq 1 -or $placeholderType -eq 3 -or $placeholderType -eq 13) {
                $keepShape = $true
            }
        } catch {
        }

        if (-not $keepShape) {
            $shape.Delete()
        }
    }
}

function Add-BulletTextBox {
    param(
        [object]$Slide,
        [string[]]$Bullets,
        [int]$FontSize = 24
    )

    $textBox = $Slide.Shapes.AddTextbox(1, 105, 150, 1110, 420)
    $textBox.Name = "Generated Body"
    $textBox.TextFrame.MarginLeft = 6
    $textBox.TextFrame.MarginRight = 6
    $textBox.TextFrame.MarginTop = 4
    $textBox.TextFrame.MarginBottom = 4
    $textBox.TextFrame.AutoSize = 0
    $textBox.TextFrame.WordWrap = -1
    $textBox.TextFrame.TextRange.Text = ($Bullets -join "`r")

     for ($paragraphIndex = 1; $paragraphIndex -le $Bullets.Count; $paragraphIndex++) {
         $paragraph = $textBox.TextFrame.TextRange.Paragraphs($paragraphIndex)
         $paragraph.Font.Name = "Aptos"
         $paragraph.Font.Size = $FontSize
        $paragraph.IndentLevel = 1
         $paragraph.ParagraphFormat.Bullet.Visible = -1
         $paragraph.ParagraphFormat.SpaceAfter = 10
     }

    return $textBox
}

function Update-TitleSlide {
    param(
        [object]$Slide,
        [hashtable]$SlideData
    )

    $textShapes = @()
    foreach ($shape in $Slide.Shapes) {
        $placeholderType = $null
        try {
            $placeholderType = $shape.PlaceholderFormat.Type
        } catch {
        }

        if ($placeholderType -eq 13) {
            continue
        }

        try {
            if ($shape.HasTextFrame -and $shape.TextFrame.HasText) {
                $textShapes += $shape
            }
        } catch {
        }
    }

    if ($textShapes.Count -lt 3) {
        throw "Unexpected title slide structure in template."
    }

    $orderedShapes = $textShapes | Sort-Object Top
    $mainShape = $orderedShapes[0]
    $footerShape = $orderedShapes[1]
    $dateShape = $orderedShapes[2]

    $mainShape.TextFrame.TextRange.Text = ($SlideData.TitleLines -join "`r")
    $mainShape.TextFrame.TextRange.ParagraphFormat.Alignment = 2
    $mainShape.TextFrame.TextRange.Font.Name = "Aptos"
    $mainShape.TextFrame.TextRange.Font.Italic = -1
    $mainShape.TextFrame.TextRange.Paragraphs(1).Font.Size = 30
    $mainShape.TextFrame.TextRange.Paragraphs(2).Font.Size = 30
    $mainShape.TextFrame.TextRange.Paragraphs(3).Font.Size = 22

    $footerShape.TextFrame.TextRange.Text = $SlideData.Footer
    $footerShape.TextFrame.TextRange.Font.Size = 26

    $dateShape.TextFrame.TextRange.Text = $SlideData.Date
    $dateShape.TextFrame.TextRange.Font.Size = 18

    Set-NotesText -Slide $Slide -Text $SlideData.Notes
}

function Update-BulletSlide {
    param(
        [object]$Slide,
        [hashtable]$SlideData
    )

    Reset-ContentSlide -Slide $Slide

    $titleShape = Get-TitleShape -Slide $Slide
    $titleShape.TextFrame.TextRange.Text = $SlideData.Title
    $titleShape.TextFrame.TextRange.Font.Name = "Aptos Display"
    $titleShape.TextFrame.TextRange.Font.Bold = -1
    $titleShape.TextFrame.TextRange.Font.Size = 28
    $titleShape.Width = 1040

    $fontSize = 24
    if ($SlideData.ContainsKey("FontSize")) {
        $fontSize = $SlideData.FontSize
    }

    Add-BulletTextBox -Slide $Slide -Bullets $SlideData.Bullets -FontSize $fontSize | Out-Null
    Set-NotesText -Slide $Slide -Text $SlideData.Notes
}

$slideDeck = @(
    @{
        Kind = "Title"
        TitleLines = @(
            "From Weak Labels to Future Episodes:",
            "Adapting Swift Hydra to Financial Time Series",
            "Dat Le"
        )
        Footer = "AlgoGators Investment Fund"
        Date = "April 2026"
        Notes = "Thanks everyone. Today I'm going to walk you through what happened when we tried to take Swift Hydra - a generative anomaly detection framework that does really well on tabular benchmarks - and apply it to financial time series. The short version: it didn't transfer, and the reason it didn't transfer turned out to be the most interesting result in the paper."
    }
    @{
        Kind = "Bullets"
        Title = "Why Anomaly Detection in Markets Is Hard"
        Bullets = @(
            "An anomaly in markets is not a point - it is an episode",
            "Benchmarks usually assume anomalies are a row-level feature-vector property",
            "Market stress unfolds across multiple days",
            "Whether today matters depends on the path the market takes over the next week"
        )
        FontSize = 24
        Notes = "Most anomaly detection benchmarks assume anomalies are roughly a point property of a feature vector - one weird row in a table. Financial markets break that assumption in two ways. First, the thing we actually care about - a stress episode - unfolds over multiple days. Second, whether today is the start of one depends on the path the market takes over the next week, not just on today's features. That temporal structure turns out to be the crack that everything else falls through."
    }
    @{
        Kind = "Bullets"
        Title = "Swift Hydra and Our Research Question"
        Bullets = @(
            "Generator -> hard synthetic rows -> detector retrains -> repeat",
            "Swift Hydra shows strong gains on tabular benchmarks like ADBench",
            "Research question: does this co-evolving loop transfer to financial time series?",
            "We answer that through three progressively more rigorous empirical branches"
        )
        FontSize = 24
        Notes = "Swift Hydra works like this: a generator produces hard anomalous samples near the detector's decision boundary, the detector retrains on them, and the two co-evolve. On benchmarks like ADBench this gives strong gains. Our research question was simple. If we build a serious futures dataset and plug Swift Hydra into the same pipeline, do we get the same lift? To answer that honestly, we had to go through three progressively more rigorous empirical branches - and the journey itself is the paper."
    }
    @{
        Kind = "Bullets"
        Title = "The Three-Branch Trajectory"
        Bullets = @(
            "anomaly_v2 | equity | 13.3M rows | PR-AUC 0.282 | circular label",
            "hybrid_v0 | live trading | 2,535 rows | PR-AUC 0.488 | too little data",
            "futures_episode_v0 | futures | 91,646 rows | PR-AUC 0.273 | honest target",
            "Each branch fixed a problem in the one before it"
        )
        FontSize = 22
        Notes = "Each branch fixed a problem in the one before it. The equity branch had plenty of data but the wrong label - it looked like a win, and it wasn't. The live-trading branch had an honest label but almost no data. And the futures branch is where both conditions are finally met. I'll take each one in turn because each one teaches a different lesson about what anomaly detection even means in this setting."
    }
    @{
        Kind = "Bullets"
        Title = "Branch 1: The Circularity Diagnosis"
        Bullets = @(
            "13.3 million equity-day rows with a rolling-threshold anomaly label",
            "XGBoost reached PR-AUC 0.282",
            "Swift Hydra slipped slightly to PR-AUC 0.277",
            "The label was a deterministic function of the same rolling-statistical features used as inputs",
            "Strong performance here does not imply real future-behavior learning"
        )
        FontSize = 22
        Notes = "Branch one was 13.3 million equity-day rows with a rolling-threshold anomaly label - basically: flag a day if some rolling z-score crosses a cutoff. XGBoost got PR-AUC 0.282, which sounded great. Then we added Swift Hydra on top - PR-AUC 0.277. A slight regression. When we dug in, we realized why. The label was a deterministic function of the same rolling-statistical features we were feeding the model. So a flexible enough detector can just recover the labeling rule and look excellent without learning anything about future market behavior. That's the circularity problem."
    }
    @{
        Kind = "Bullets"
        Title = "Theory: The Circular-Label Bayes Ceiling"
        Bullets = @(
            "If Y depends only on feature subset S, then eta(x) = P(Y = 1 | S)",
            "Extra inputs U carry no information once S is known",
            "More capacity and more synthetic samples can only fit noise around that ceiling",
            "The real fix is a better label, not a fancier detector"
        )
        FontSize = 24
        Notes = "This has a clean theoretical statement. If the label is conditionally independent of the rest of your inputs given the features that generate it, the Bayes-optimal score collapses to the labeling rule itself. Every bit of extra modeling capacity - and every synthetic sample - can at best fit noise around that ceiling. So the real fix wasn't a better model. It was a better label."
    }
    @{
        Kind = "Bullets"
        Title = "Branch 2: The Live-Trading Pivot"
        Bullets = @(
            "Honest label based on actual live-trading failures",
            "2,535 rows across 129 dates",
            "Logistic regression reached PR-AUC 0.488",
            "One strategy contributed almost all of the data",
            "Autocorrelation crushed the effective sample size"
        )
        FontSize = 23
        Notes = "Branch two redefined anomalies around actual live-trading failures. The label was honest this time - and a simple logistic regression got PR-AUC 0.488, the highest number in the paper. But we only had 2,535 rows across 129 dates, and one strategy contributed almost all of it. Under autocorrelation, effective sample size is much smaller than nominal, and any model more complex than logistic regression degraded. We needed a new panel."
    }
    @{
        Kind = "Bullets"
        Title = "Branch 3: The Future-Episode Label"
        Bullets = @(
            "Flag row t if the next H = 5 trading days contain a multi-day stress episode",
            "Forward criteria: drawdown > tau^dd, stress score G > tau^crit, or >= 2 elevated-stress days",
            "Thresholds are estimated strictly from prior history, symbol by symbol",
            "Panel size: 36 contracts, 2,881 trading days, 91,646 rows",
            "The label is forward-looking and not a function of current-row features"
        )
        FontSize = 22
        Notes = "The mainline label is strictly forward-looking. For each row, we ask: over the next five trading days, does this symbol experience a multi-day stress episode? We combine three conditions - forward drawdown, a composite stress score, and a count of individually elevated days. All thresholds are estimated from strictly prior history, symbol by symbol, so there's no leakage. That gave us a 91,646-row futures panel spanning 36 contracts and nine years - two orders of magnitude more temporal coverage than branch two, and a label that is provably not a function of current-row features."
    }
    @{
        Kind = "Bullets"
        Title = "Honest Baseline Results"
        Bullets = @(
            "LogReg + symbol fixed effects: PR-AUC 0.273",
            "Compact XGBoost baseline: PR-AUC 0.237",
            "Prevalence is 5.86%, so 0.273 is a 4.66x lift over base rate",
            "At 5% FPR, the queue is roughly 1-in-3 precision",
            "The biggest gain came from contract fixed effects, not extra feature complexity"
        )
        FontSize = 22
        Notes = "Now for what actually worked. A compact logistic regression with contract fixed effects - meaning one-hot indicators for symbol - reaches test PR-AUC 0.273. Against a 5.86 percent base rate, that's a 4.66-times prevalence-normalized lift. Operationally, it converts a one-in-seventeen base rate into a flagged queue with roughly one-in-three precision at a five-percent false-positive rate. The most interesting finding here is that the biggest jump didn't come from adding features - it came from contract fixed effects. Which tells us baseline stress propensity differs more across instruments than within an instrument over time."
    }
    @{
        Kind = "Bullets"
        Title = "Swift Hydra Transfer: Three Failed Variants"
        Bullets = @(
            "Tabular frontier to beat: PR-AUC 0.273",
            "Row-level RL synthesis: 0.227",
            "GRU sequence baseline: 0.192",
            "Sequence-conditioned RL: 0.173",
            "Meta-reweighting, influence-guided interpolation, and stacking stayed non-positive"
        )
        FontSize = 21
        Notes = "Now, against that honest 0.273 frontier, we ran Swift Hydra three different ways. Row-level RL synthesis - the original recipe - got 0.227, below even the plain XGBoost baseline. We said okay, the label is an episode property, let's synthesize precursor windows instead. Sequence-conditioned RL got 0.173, worse again. We tried meta-reweighting, influence-guided interpolation, and simple stacking - all non-positive relative to the anchor. So this is not a tuning story. Across every variant we tried, synthetic hardening did not improve the detector."
    }
    @{
        Kind = "Bullets"
        Title = "Why It Failed: Structural Diagnosis"
        Bullets = @(
            "Object mismatch: the generator outputs rows, but the label is a future-path property",
            "Mass limitation: synthetic influence is bounded by B * m / (n + m)",
            "In our setting, m << n, so synthetic mass stays small",
            "Reward misspecification: detector evasion is not downstream validation gain",
            "Need a bilevel objective aligned to held-out PR-AUC"
        )
        FontSize = 20
        Notes = "We think the failure traces to three things, and they compound. First, object mismatch - Swift Hydra generates individual feature vectors, but our label is a property of what happens over the next five days. A synthetic row that fools the current detector isn't necessarily a plausible precursor to an actual stress episode. Second, mass limitation - augmenting n real samples with m synthetic ones bounds their influence by B times m over n plus m, and for us m is tiny compared to n. Third, the reward is wrong. Detector evasion is the wrong proxy for downstream validation gain. The correct generator objective is a bilevel problem - produce precursor windows that maximize held-out PR-AUC after retraining - which is what we formalize in Equations 1 and 2."
    }
    @{
        Kind = "Bullets"
        Title = "Future Work and Takeaways"
        Bullets = @(
            "Sequence-native conditional generation with diffusion models or TimeGAN",
            "Utility-aligned generator rewards built around the bilevel objective",
            "Targeted seeds from influence functions or TracIn",
            "Richer state with term structure, intraday stress, and implied volatility",
            "Takeaway: synthetic objects must match the prediction target"
        )
        FontSize = 20
        Notes = "Where does this go next? The key open problem is instantiating the bilevel objective - rewarding the generator for real downstream gain on unseen stress episodes, not for fooling the current detector. Conditional diffusion models and trajectory-matching losses are the natural next candidates. The bigger takeaway, and what I want you to leave with, is this: before you reach for a fancier generator, check that the synthetic object you're producing actually matches the prediction target. When the label is a future-path property, a single row isn't the right unit - and no amount of Swift-Hydra-style hardening fixes that. Thank you - happy to take questions."
    }
)

if (-not (Test-Path -LiteralPath $TemplatePath)) {
    throw "Template file not found: $TemplatePath"
}

$outputDirectory = Split-Path -Parent $OutputPath
if (-not (Test-Path -LiteralPath $outputDirectory)) {
    New-Item -ItemType Directory -Path $outputDirectory | Out-Null
}

Copy-Item -LiteralPath $TemplatePath -Destination $OutputPath -Force

$powerPoint = $null
$presentation = $null

try {
    $powerPoint = New-Object -ComObject PowerPoint.Application
    $presentation = $powerPoint.Presentations.Open($OutputPath, $false, $false, $false)

    while ($presentation.Slides.Count -gt $slideDeck.Count) {
        $presentation.Slides.Item($presentation.Slides.Count).Delete()
    }

    if ($presentation.Slides.Count -lt $slideDeck.Count) {
        throw "Template has fewer slides than expected."
    }

    for ($slideIndex = 1; $slideIndex -le $slideDeck.Count; $slideIndex++) {
        $slide = $presentation.Slides.Item($slideIndex)
        $slideData = $slideDeck[$slideIndex - 1]

        if ($slideData.Kind -eq "Title") {
            Update-TitleSlide -Slide $slide -SlideData $slideData
        } else {
            Update-BulletSlide -Slide $slide -SlideData $slideData
        }
    }

    $presentation.Save()
    Write-Output "Saved presentation to $OutputPath"
} finally {
    if ($presentation -ne $null) {
        $presentation.Close()
    }
    if ($powerPoint -ne $null) {
        $powerPoint.Quit()
    }
}
