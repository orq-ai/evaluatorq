(() => {
  // USD per 1k tokens; provenance and date in docs/guides/red-teaming.md.
  const modelPrices = {
    "claude-opus-5": { input: 0.005, output: 0.025 },
    "claude-sonnet-5": { input: 0.003, output: 0.015 },
    "claude-haiku-4-5": { input: 0.001, output: 0.005 },
    "gpt-5-mini": { input: 0.00025, output: 0.002 },
  };

  const CACHE_READ_MULTIPLIER = 0.1;
  const JUDGE_OUTPUT_TOKENS = 200;

  const usdFormatter = new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 6,
  });
  const formatUsd = (value) => usdFormatter.format(value);

  const initCostCalculators = () => {
    document.querySelectorAll("[data-cost-calculator]").forEach((calculator) => {
      if (calculator.dataset.initialized) return;

      const input = (name) => calculator.elements.namedItem(name);
      const model = input("model");
      const priceFields = ["input-price", "output-price"].map(input);
      const total = calculator.querySelector("[data-cost-total]");
      const breakdown = calculator.querySelector("[data-cost-breakdown]");
      const fields = [
        "fixed-calls",
        "attacks",
        "turns",
        "calls-per-turn",
        "judge-calls",
        "tokens-per-side",
        "cache-hit",
      ].map(input);

      // Fail loudly and skip this instance rather than throwing on every keystroke:
      // the markup is hand-authored HTML inside Markdown, so a dropped element is
      // a realistic docs edit and a silent dead widget is worse than a console line.
      if (!model || !total || !breakdown || [...priceFields, ...fields].some((f) => !f)) {
        console.warn("cost-calculator: missing required fields, skipping instance", calculator);
        return;
      }
      calculator.dataset.initialized = "true";

      const num = (field) => Math.max(0, Number(field.value) || 0);

      const calculate = () => {
        const [fixedCalls, attacks, turns, callsPerTurn, judgeCalls, tokensPerSide, cacheHitPct] =
          fields.map(num);
        const [inputPrice, outputPrice] = priceFields.map(num);
        const cacheHit = Math.min(1, cacheHitPct / 100);
        const cachedInputPrice =
          inputPrice * (1 - cacheHit + cacheHit * CACHE_READ_MULTIPLIER);

        // Turn calls read the transcript so far and append a block to it, seeded
        // with one block so the first call is not free. Judge calls read the
        // finished transcript without extending it for each other.
        const turnCalls = turns * callsPerTurn;
        let transcript = tokensPerSide;
        let turnInput = 0;
        for (let call = 0; call < turnCalls; call += 1) {
          turnInput += transcript;
          transcript += tokensPerSide;
        }
        const attackInput = turnInput + judgeCalls * transcript;
        const attackOutput = turnCalls * tokensPerSide + judgeCalls * JUDGE_OUTPUT_TOKENS;

        // Setup calls share no prefix with the attack transcript, so no cache read.
        const fixedInput = fixedCalls * tokensPerSide;
        const fixedOutput = fixedCalls * tokensPerSide;
        const inputTokens = fixedInput + attacks * attackInput;
        const outputTokens = fixedOutput + attacks * attackOutput;

        const calls = fixedCalls + attacks * (turnCalls + judgeCalls);
        const cost =
          (fixedInput / 1000) * inputPrice +
          ((attacks * attackInput) / 1000) * cachedInputPrice +
          (outputTokens / 1000) * outputPrice;

        total.textContent = formatUsd(cost);
        breakdown.textContent =
          `${calls.toLocaleString()} calls · ` +
          `${inputTokens.toLocaleString()} input tokens (${Math.round(cacheHit * 100)}% cached) · ` +
          `${outputTokens.toLocaleString()} output tokens`;
      };

      calculator.querySelectorAll("input").forEach((field) => {
        field.addEventListener("input", () => {
          if (priceFields.includes(field)) model.value = "custom";
          calculate();
        });
      });
      calculator.addEventListener("submit", (event) => event.preventDefault());
      model.addEventListener("change", () => {
        const price = modelPrices[model.value];
        if (price) {
          priceFields[0].value = price.input;
          priceFields[1].value = price.output;
        }
        calculate();
      });
      calculate();
    });
  };

  initCostCalculators();
  if (typeof document$ !== "undefined") document$.subscribe(initCostCalculators);
})();
