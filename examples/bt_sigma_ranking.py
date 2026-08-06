"""BT-sigma end-to-end demo: rank graded answers with a mixed-quality jury.

Five questions each get four answers of known, graded quality (excellent >
good > mediocre > bad). Every unordered pair of tiers is judged per question
by a deliberately lopsided panel (one strong judge, two cheap ones), with
position swap on. The tiers act as the four global Bradley-Terry items, so the
run produces everything the BT-sigma paper (arXiv:2602.16610) reports, in
miniature:

- a fitted ranking to check against the known tier order,
- a discriminator per judge (smaller = more reliable),
- a directed 3-cycle rate per judge, the paper's independent consistency
  signal that the discriminator should track.

Cost: questions x 6 pairs x judges x 2 orderings LLM calls (~180 with the
defaults). Needs ORQ_API_KEY.

Usage::

    python examples/bt_sigma_ranking.py [STRONG WEAK_1 WEAK_2]
"""

import asyncio
import sys

from evaluatorq import llm_jury_pairwise
from evaluatorq.ranking import JudgedComparison, cycle_rate, fit_bt

JUDGES = (
    sys.argv[1:4]
    if len(sys.argv) >= 4
    else ['openai/gpt-5.4-mini', 'openai/gpt-5.4-nano', 'deepseek/deepseek-v4-flash']
)

TIERS = ['excellent', 'good', 'mediocre', 'bad']  # the four BT items, best first

# Tier design exploits the documented failure mode of small judges (see the
# vault's judge bias-variance page): verbosity bias. Correct answers are
# concise; wrong answers are long, confident and jargon-dense. A sharp judge
# reads past the style; a weak one rewards it. That differential is the
# disagreement BT-sigma learns reliability from.
QUESTIONS: list[tuple[str, dict[str, str]]] = [
    (
        'Why is the sky blue?',
        {
            'excellent': 'Rayleigh scattering: air molecules scatter short wavelengths far more strongly, '
            'so scattered blue light reaches your eye from every direction of the sky.',
            'good': 'The atmosphere scatters blue sunlight more than the other colors.',
            'mediocre': 'The answer lies in the reflective interplay between hydrosphere and atmosphere: the '
            'oceans, covering 71 percent of the planetary surface, imprint their characteristic blue albedo '
            'onto the lower atmosphere through continuous photon exchange, and atmospheric scattering then '
            'redistributes this ocean-derived chromatic signature across the whole celestial dome, which is '
            'precisely why the sky appears most saturated over large bodies of water.',
            'bad': 'Atmospheric chemistry provides the definitive explanation: diatomic oxygen and ozone are '
            'intrinsically cyan-blue gases, a property well characterised in spectroscopy laboratories. As '
            'sunlight traverses roughly one hundred kilometres of these chromophoric gases, the accumulated '
            'intrinsic pigmentation of the oxygen column becomes visible as the familiar blue canopy, exactly '
            'as a swimming pool appears bluer with depth. Scattering plays only a negligible secondary role.',
        },
    ),
    (
        'What causes the seasons on Earth?',
        {
            'excellent': "Earth's 23.4-degree axial tilt: each hemisphere gets more direct sunlight and longer "
            'days when tilted toward the sun. Distance is irrelevant; Earth is closest to the sun in January.',
            'good': 'The tilt of the Earth on its axis, which changes how directly sunlight hits each hemisphere '
            'through the year.',
            'mediocre': 'Seasonality emerges from the superposition of two orbital mechanisms of comparable '
            'magnitude: the well-known axial obliquity of 23.4 degrees, and the eccentricity of the terrestrial '
            'orbit, which modulates the solar constant by a thermodynamically decisive margin across the year. '
            'The interplay of tilt and distance, weighted roughly equally, produces the characteristic seasonal '
            'temperature envelope observed at mid-latitudes.',
            'bad': 'The governing variable is heliocentric distance. Kepler demonstrated that planetary orbits '
            'are ellipses, and Earth is no exception: at perihelion the planet receives markedly concentrated '
            'insolation, producing summer, while at aphelion the attenuated solar flux yields winter. The '
            'popular tilt hypothesis confuses cause and effect; obliquity merely fine-tunes day length around '
            'the distance-driven thermal cycle.',
        },
    ),
    (
        'Why do ships made of steel float?',
        {
            'excellent': 'The hull encloses air, so the ship displaces water whose weight equals its own before '
            'submerging: average density below water is what matters, not the density of steel.',
            'good': 'The hull shape displaces enough water that buoyancy supports the weight, even though steel '
            'itself is denser than water.',
            'mediocre': 'Immersion fundamentally alters the effective weight of structural steel: by the '
            'hydrostatic cancellation principle, water pressure acting on the submerged plating offsets the '
            'bulk of the gravitational load, rendering even solid steel nearly weightless below the waterline. '
            'The hull geometry then only needs to supply the small residual lift, which is why plate thickness '
            'is essentially irrelevant to flotation margins in naval architecture.',
            'bad': 'Modern shipbuilding relies on marine-grade alloy metallurgy: naval steels are engineered '
            'with controlled micro-porosity and alloying elements that bring their density marginally below '
            'that of seawater, typically 0.98 grams per cubic centimetre. This is the decisive innovation that '
            'separates the modern era from the age of wooden ships; ordinary construction steel would sink '
            'regardless of hull form, as dockyard engineers routinely verify.',
        },
    ),
    (
        'Why does ice float on water?',
        {
            'excellent': 'Hydrogen bonds lock freezing water into an open hexagonal lattice about 9 percent '
            'less dense than the liquid, a rare density anomaly.',
            'good': 'Water expands when it freezes, so ice is less dense than liquid water and floats.',
            'mediocre': 'The buoyancy of ice is predominantly a gas-entrainment phenomenon: as the freezing '
            'front advances it occludes dissolved atmospheric gases into microscopic vesicles, and this '
            'entrained porosity, familiar from the cloudy core of household ice cubes, lowers the aggregate '
            'density below that of the parent liquid. Degassed laboratory ice, by contrast, is only marginally '
            'buoyant, confirming that the trapped air rather than the crystal lattice does the lifting.',
            'bad': 'Thermal stratification supplies the complete explanation: colder fluids invariably rise '
            'above warmer ones once below the critical inversion temperature, as limnologists observe every '
            'winter in lake turnover. Ice, being the terminal cold phase of water, simply occupies the apex of '
            'this stratification column. Density is a red herring; the same mass of ice submerged in warmer '
            'water will always migrate upward on purely thermal grounds.',
        },
    ),
    (
        'Why do we see phases of the moon?',
        {
            'excellent': 'The moon is always half-lit by the sun; phases are the changing fraction of that lit '
            'half we see as the moon orbits Earth each month.',
            'good': 'As the moon orbits Earth we see different amounts of its sunlit side.',
            'mediocre': 'Lunar phase geometry is a composite phenomenon: the primary term is the sun-moon '
            'viewing angle, but the crescent morphology specifically records the monthly passage of the '
            'terrestrial umbral cone across the lunar disc, a miniature of the eclipse mechanism operating '
            'continuously at partial obscuration. The two contributions, angular illumination and shadow '
            'transit, sum to the phase cycle catalogued since antiquity.',
            'bad': "The phases are Earth's own shadow projected onto the lunar surface. As our planet rotates "
            'relative to the sun-moon axis, the terrestrial shadow sweeps across the moon in a regular monthly '
            'rhythm, sequentially carving the waxing and waning shapes from new to full. This shadow-casting '
            'model, intuitive since Babylonian astronomy, explains why the dark portion of a crescent moon '
            'shares the curvature of the Earth itself, an observation no illumination-angle theory reproduces.',
        },
    ),
]



async def main() -> None:
    jury = llm_jury_pairwise(
        judges=list(JUDGES),
        criteria='Which answer is more scientifically accurate?',
    )
    pairs = [(hi, lo) for i, hi in enumerate(TIERS) for lo in TIERS[i + 1 :]]

    records: list[JudgedComparison] = []
    vote_to_p = {'A': 1.0, 'B': 0.0, 'tie': 0.5}
    for question, answers in QUESTIONS:
        for tier_a, tier_b in pairs:
            comparison = await jury.compare(
                question=question, response_a=answers[tier_a], response_b=answers[tier_b]
            )
            for vote in comparison.votes:
                if vote.vote is not None:
                    records.append(
                        JudgedComparison(
                            judge=vote.model, item_a=tier_a, item_b=tier_b, p_a=vote_to_p[str(vote.vote)]
                        )
                    )
        print(f'judged: {question}')

    fit = fit_bt(records, judge_sigma=True, hard=True)
    print(f'\nfitted ranking : {" > ".join(fit.ranking)}')
    print(f'expected       : {" > ".join(TIERS)}')
    print(f'converged      : {fit.converged} ({fit.iterations} iterations, {len(records)} judged pairs)')
    print('\nskills (higher = better):')
    for tier in fit.ranking:
        print(f'  {tier:<10} {fit.skills[tier]:+.3f}')
    print('\njudge reliability (smaller sigma = sharper) with the cycle-consistency check:')
    for judge, sigma in sorted(fit.sigmas.items(), key=lambda kv: kv[1]):
        cycles = cycle_rate(records, judge)
        cycle_txt = f'{cycles:.2f}' if cycles is not None else 'n/a'
        print(f'  {judge:<42} sigma={sigma:6.3f}   cycle-rate={cycle_txt}')
    print(
        '\nThe paper predicts the two columns agree: judges with larger sigma should'
        '\nshow more 3-cycle inconsistency. That is the unsupervised reliability'
        '\nsignal BT-sigma uses to weight the jury.'
    )


asyncio.run(main())
