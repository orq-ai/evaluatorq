(() => {
  // USD per 1k tokens; provenance and date in docs/guides/red-teaming.md.
  const modelPrices = {
    "claude-opus-5": { input: 0.005, output: 0.025 },
    "claude-sonnet-5": { input: 0.003, output: 0.015 },
    "claude-haiku-4-5": { input: 0.001, output: 0.005 },
    "gpt-5-mini": { input: 0.00025, output: 0.002 },
  };

  // Fixed shape of a dynamic/hybrid attack, and the ballpark assumptions the
  // guide states next to the widget. Not user-editable: they are what the
  // pipeline does, not knobs.
  const CALLS_PER_TURN = 2; // adversarial generation + target call
  const JUDGE_CALLS = 1; // once after the turn loop, not per turn
  const TOKENS_PER_SIDE = 1000;
  const JUDGE_OUTPUT_TOKENS = 200;
  const CACHE_HIT = 0.9;
  const CACHE_READ_MULTIPLIER = 0.1;

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
      const fields = ["fixed-calls", "attacks", "turns"].map(input);
      const customPrices = calculator.querySelector("[data-custom-prices]");

      // Fail loudly and skip this instance rather than throwing on every keystroke:
      // the markup is hand-authored HTML inside Markdown, so a dropped element is
      // a realistic docs edit and a silent dead widget is worse than a console line.
      if (
        !model ||
        !total ||
        !breakdown ||
        !customPrices ||
        [...priceFields, ...fields].some((f) => !f)
      ) {
        console.warn("cost-calculator: missing required fields, skipping instance", calculator);
        return;
      }
      calculator.dataset.initialized = "true";

      const num = (field) => Math.max(0, Number(field.value) || 0);

      const calculate = () => {
        const [fixedCalls, attacks, turns] = fields.map(num);
        const [inputPrice, outputPrice] = priceFields.map(num);
        const cachedInputPrice =
          inputPrice * (1 - CACHE_HIT + CACHE_HIT * CACHE_READ_MULTIPLIER);

        // Turn calls read the transcript so far and append a block to it, seeded
        // with one block so the first call is not free. Judge calls read the
        // finished transcript without extending it for each other.
        const turnCalls = turns * CALLS_PER_TURN;
        let transcript = TOKENS_PER_SIDE;
        let turnInput = 0;
        for (let call = 0; call < turnCalls; call += 1) {
          turnInput += transcript;
          transcript += TOKENS_PER_SIDE;
        }
        const attackInput = turnInput + JUDGE_CALLS * transcript;
        const attackOutput = turnCalls * TOKENS_PER_SIDE + JUDGE_CALLS * JUDGE_OUTPUT_TOKENS;

        // Setup calls share no prefix with the attack transcript, so no cache read.
        const fixedInput = fixedCalls * TOKENS_PER_SIDE;
        const fixedOutput = fixedCalls * TOKENS_PER_SIDE;
        const inputTokens = fixedInput + attacks * attackInput;
        const outputTokens = fixedOutput + attacks * attackOutput;

        const calls = fixedCalls + attacks * (turnCalls + JUDGE_CALLS);
        const cost =
          (fixedInput / 1000) * inputPrice +
          ((attacks * attackInput) / 1000) * cachedInputPrice +
          (outputTokens / 1000) * outputPrice;

        total.textContent = formatUsd(cost);
        breakdown.textContent =
          `${calls.toLocaleString()} calls · ` +
          `${inputTokens.toLocaleString()} input tokens (${Math.round(CACHE_HIT * 100)}% cached) · ` +
          `${outputTokens.toLocaleString()} output tokens`;
      };

      // Prices are a consequence of the model choice, not a knob — reveal the
      // two inputs only when there is no catalogue price to consequence from.
      const syncModel = () => {
        const price = modelPrices[model.value];
        if (price) {
          priceFields[0].value = price.input;
          priceFields[1].value = price.output;
        }
        customPrices.hidden = Boolean(price);
      };

      calculator.querySelectorAll("input").forEach((field) => {
        field.addEventListener("input", calculate);
      });
      calculator.addEventListener("submit", (event) => event.preventDefault());
      model.addEventListener("change", () => {
        syncModel();
        calculate();
      });
      syncModel();
      calculate();
    });
  };

  initCostCalculators();
  if (typeof document$ !== "undefined") document$.subscribe(initCostCalculators);
})();
