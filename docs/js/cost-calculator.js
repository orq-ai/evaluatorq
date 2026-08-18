(() => {
  // Published list prices in USD per 1k tokens, as of 2026-08-18.
  // Anthropic: anthropic.com/pricing. gpt-5-mini: the Orq /v2/models payload,
  // which is what common/model_catalogue.py reads at runtime.
  // These are a dated snapshot for planning, not a live feed — re-check before
  // quoting them at anyone.
  const modelPrices = {
    "claude-opus-5": { input: 0.005, output: 0.025 },
    "claude-sonnet-5": { input: 0.003, output: 0.015 },
    "claude-haiku-4-5": { input: 0.001, output: 0.005 },
    "gpt-5-mini": { input: 0.00025, output: 0.002 },
  };

  // Cache reads bill at ~0.1x the base input price across the providers above.
  const CACHE_READ_MULTIPLIER = 0.1;

  const usdFormatter = new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 4,
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
        const effectiveInputPrice =
          inputPrice * (1 - cacheHit + cacheHit * CACHE_READ_MULTIPLIER);

        // Every call reads the transcript so far and appends tokensPerSide to it.
        // Seeded with one block so the first call is not free (system prompt +
        // attack objective). Fixed setup calls do not share the attack transcript,
        // so each is priced as a single seed-sized exchange.
        let inputTokens = fixedCalls * tokensPerSide;
        let outputTokens = fixedCalls * tokensPerSide;
        const callsPerAttack = turns * callsPerTurn + judgeCalls;
        for (let attack = 0; attack < attacks; attack += 1) {
          let transcript = tokensPerSide;
          for (let call = 0; call < callsPerAttack; call += 1) {
            inputTokens += transcript;
            outputTokens += tokensPerSide;
            transcript += tokensPerSide;
          }
        }

        const calls = fixedCalls + attacks * callsPerAttack;
        const cost =
          (inputTokens / 1000) * effectiveInputPrice + (outputTokens / 1000) * outputPrice;

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
