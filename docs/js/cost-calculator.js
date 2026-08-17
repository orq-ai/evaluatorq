(() => {
  const tierPrices = {
    frontier: 0.01,
    balanced: 0.002,
    cheap: 0.0004,
  };

  const formatUsd = (value) =>
    new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 6,
    }).format(value);

  const initCostCalculators = () => {
    document.querySelectorAll("[data-cost-calculator]").forEach((calculator) => {
      if (calculator.dataset.initialized) return;
      calculator.dataset.initialized = "true";

      const input = (name) => calculator.elements.namedItem(name);
      const tier = input("model-tier");
      const callCostField = input("call-cost");
      const total = calculator.querySelector("[data-cost-total]");
      const breakdown = calculator.querySelector("[data-cost-breakdown]");

      const calculate = () => {
        const fixedCalls = Math.max(0, Number(input("fixed-calls").value) || 0);
        const attacks = Math.max(0, Number(input("attacks").value) || 0);
        const turns = Math.max(0, Number(input("turns").value) || 0);
        const callCost = Math.max(0, Number(callCostField.value) || 0);
        const variableCalls = attacks * turns * 2;
        const calls = fixedCalls + variableCalls;

        total.textContent = formatUsd(calls * callCost);
        breakdown.textContent = `${calls.toLocaleString()} calls at ${formatUsd(callCost)}/call = ${fixedCalls.toLocaleString()} fixed + ${variableCalls.toLocaleString()} attack/turn`;
      };

      calculator.querySelectorAll("input").forEach((field) => {
        field.addEventListener("input", calculate);
      });
      callCostField.addEventListener("input", () => {
        tier.value = "custom";
        calculate();
      });
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
