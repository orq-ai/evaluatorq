(() => {
  const tierPrices = {
    frontier: 0.01,
    balanced: 0.002,
    cheap: 0.0004,
  };

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
      const tier = input("model-tier");
      const callCostField = input("call-cost");
      const total = calculator.querySelector("[data-cost-total]");
      const breakdown = calculator.querySelector("[data-cost-breakdown]");
      const fields = ["fixed-calls", "attacks", "turns", "calls-per-turn", "judge-calls"].map(input);

      // Fail loudly and skip this instance rather than throwing on every keystroke:
      // the markup is hand-authored HTML inside Markdown, so a dropped element is
      // a realistic docs edit and a silent dead widget is worse than a console line.
      if (!tier || !callCostField || !total || !breakdown || fields.some((f) => !f)) {
        console.warn("cost-calculator: missing required fields, skipping instance", calculator);
        return;
      }
      calculator.dataset.initialized = "true";

      const num = (field) => Math.max(0, Number(field.value) || 0);

      const calculate = () => {
        const [fixedCalls, attacks, turns, callsPerTurn, judgeCalls] = fields.map(num);
        const callCost = Math.max(0, Number(callCostField.value) || 0);
        const variableCalls = attacks * (turns * callsPerTurn + judgeCalls);
        const calls = fixedCalls + variableCalls;

        total.textContent = formatUsd(calls * callCost);
        breakdown.textContent = `${calls.toLocaleString()} calls at ${formatUsd(callCost)}/call = ${fixedCalls.toLocaleString()} fixed + ${variableCalls.toLocaleString()} from attacks`;
      };

      calculator.querySelectorAll("input").forEach((field) => {
        field.addEventListener("input", () => {
          if (field === callCostField) tier.value = "custom";
          calculate();
        });
      });
      calculator.addEventListener("submit", (event) => event.preventDefault());
      tier.addEventListener("change", () => {
        if (tierPrices[tier.value] !== undefined) {
          callCostField.value = tierPrices[tier.value];
        }
        calculate();
      });
      calculate();
    });
  };

  initCostCalculators();
  if (typeof document$ !== "undefined") document$.subscribe(initCostCalculators);
})();
