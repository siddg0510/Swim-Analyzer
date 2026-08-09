"""
Expert swimming analysis prompts for the Gemini API.

Each prompt is carefully engineered to make Gemini act as an elite-level
swimming coach / biomechanics analyst. The prompts request structured JSON
output so the app can reliably parse and display the results.

IMPORTANT: These prompts include real biomechanical terminology and
coaching concepts from competitive swimming. The analysis quality depends
on Gemini's multimodal video understanding — it will identify technique
elements it can visually observe and flag anything it can't confidently
assess rather than guessing.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# System instruction — shared across all swimming analysis calls
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION = """You are an elite swimming biomechanics analyst and certified
World Aquatics (FINA) Level 3 coach with 20+ years of experience analyzing
Olympic and World Championship race footage. You have deep expertise in all
four competitive strokes (freestyle/front crawl, backstroke, breaststroke,
butterfly) and individual medley (IM) racing.

Your analysis style:
- Be SPECIFIC: reference exact body parts, angles, timing, and positions
- Be ACTIONABLE: every observation should connect to what the swimmer can DO differently
- Be HONEST: if the video quality, angle, or visibility doesn't allow you to
  assess something, say so — never fabricate an observation
- Use proper swimming terminology (catch, pull, recovery, entry, kick timing,
  body roll, streamline, breakout, DPS, etc.)
- Reference elite swimmers as benchmarks when relevant (e.g. "Pan Zhanle's
  high elbow catch", "Ledecky's 6-beat kick timing")
- Prioritize feedback by IMPACT: what change will produce the biggest
  improvement in race time?

CRITICAL: Always respond with valid JSON matching the schema requested.
Do NOT wrap your response in markdown code fences. Output raw JSON only."""


# ---------------------------------------------------------------------------
# Technique Analysis
# ---------------------------------------------------------------------------
TECHNIQUE_ANALYSIS_PROMPT = """Analyze this swimming video in detail. The swimmer is performing
{stroke} over a {distance}m {pool_type} race.

Evaluate EVERY aspect you can observe from this footage:

1. BODY POSITION: Head alignment, hip height, body roll/undulation, streamline quality
2. ARM MECHANICS: Entry angle, catch position (high elbow vs dropped), pull path,
   hand acceleration through the stroke, recovery (over-water phase), hand exit
3. KICK: Type (flutter/dolphin/whip), frequency, amplitude, timing with arm stroke,
   knee bend degree, ankle flexibility
4. BREATHING: Pattern (unilateral/bilateral/every stroke), head rotation degree,
   timing relative to arm cycle, impact on body alignment
5. TIMING & COORDINATION: Arm-arm coordination, arm-kick timing, stroke tempo consistency
6. START (if visible): Reaction, dive angle, entry splash, underwater streamline,
   dolphin kick count and power, breakout timing
7. TURN (if visible): Approach wall, flip/open turn execution, push-off power,
   underwater streamline and kicks, breakout distance
8. FINISH (if visible): Final stroke timing, touch accuracy, deceleration pattern

For each element, rate it: "excellent", "good", "needs_improvement", or "critical_issue".
Only rate what you can ACTUALLY SEE in the video.

Respond with this exact JSON structure:
{{
    "stroke_identified": "<stroke name>",
    "stroke_confidence": <0.0-1.0>,
    "overall_rating": "<excellent|good|needs_improvement|critical_issue>",
    "technique_elements": [
        {{
            "category": "<body_position|arm_mechanics|kick|breathing|timing|start|turn|finish>",
            "element": "<specific technique element name>",
            "rating": "<excellent|good|needs_improvement|critical_issue>",
            "observation": "<what you see in the video — specific, factual>",
            "recommendation": "<what to change and WHY it will help — actionable>",
            "elite_reference": "<optional: which elite swimmer does this well and how>",
            "timestamp_hint": "<optional: approximate time in video, e.g. '0:03-0:05'>"
        }}
    ],
    "top_3_priorities": [
        {{
            "priority": 1,
            "element": "<which technique element>",
            "expected_impact": "<how much time/efficiency this could gain>",
            "drill": "<specific drill to practice this>"
        }}
    ],
    "video_quality_notes": "<any issues with camera angle, visibility, etc. that limited analysis>"
}}"""


# ---------------------------------------------------------------------------
# Elite Comparison
# ---------------------------------------------------------------------------
ELITE_COMPARISON_PROMPT = """Compare this swimmer's {stroke} technique against the elite benchmark
of {elite_name} ({elite_country}), who {elite_achievement}.

Known elite technique characteristics of {elite_name}:
{elite_technique_description}

Known elite performance data:
{elite_performance_data}

The user's measured metrics from CV analysis:
- Average velocity: {user_velocity} m/s
- Stroke rate: {user_stroke_rate} cycles/min
- Stroke length: {user_stroke_length} m/cycle
- Event: {distance}m {stroke}

Analyze the video and compare against {elite_name}'s known technique in these areas:
1. What technique elements match or approach elite level?
2. What are the BIGGEST differences from the elite model?
3. Which differences are most addressable (technique vs physical limitations)?
4. Specific metrics comparison (stroke rate, stroke length, velocity per phase)

Respond with this JSON:
{{
    "comparison_athlete": "{elite_name}",
    "overall_similarity_pct": <0-100>,
    "strengths_matching_elite": [
        {{
            "element": "<what the swimmer does well>",
            "similarity": "<how close to elite level>",
            "detail": "<specific observation>"
        }}
    ],
    "key_differences": [
        {{
            "element": "<technique element>",
            "user_observation": "<what the user does>",
            "elite_model": "<what {elite_name} does>",
            "gap_severity": "<minor|moderate|major>",
            "addressable": true,
            "how_to_close_gap": "<specific recommendation>"
        }}
    ],
    "metrics_comparison": {{
        "velocity_gap_pct": <percentage difference>,
        "stroke_rate_comparison": "<faster/slower/similar and by how much>",
        "stroke_length_comparison": "<longer/shorter/similar and by how much>",
        "efficiency_note": "<stroke length × stroke rate balance analysis>"
    }},
    "realistic_targets": [
        {{
            "metric": "<which metric>",
            "current": "<current value>",
            "target": "<achievable target based on training>",
            "timeframe": "<estimated weeks/months to reach>",
            "method": "<how to get there>"
        }}
    ]
}}"""


# ---------------------------------------------------------------------------
# Improvement Plan
# ---------------------------------------------------------------------------
IMPROVEMENT_PLAN_PROMPT = """Based on this swimming video analysis, create a comprehensive,
personalized improvement plan for this {stroke} swimmer targeting {distance}m events.

Technique analysis findings:
{technique_findings}

Current performance metrics:
- Velocity: {user_velocity} m/s
- Stroke rate: {user_stroke_rate} cycles/min
- Stroke length: {user_stroke_length} m/cycle
- Start phase (0-15m): {start_time} s
- Turn time: {turn_time} s

The swimmer's goal: {user_goal}

Create a structured improvement plan that a competitive swimmer can follow.
Reference real drills used by Olympic coaches (e.g. Bob Bowman, Brett Hawke,
David Marsh, etc.) where applicable.

Respond with this JSON:
{{
    "summary": "<1-2 sentence overview of the plan's focus>",
    "phase_1_immediate": {{
        "duration_weeks": <2-4>,
        "focus": "<main technical focus>",
        "drills": [
            {{
                "name": "<drill name>",
                "purpose": "<what it fixes>",
                "description": "<how to perform it>",
                "sets_reps": "<recommended volume, e.g. 4x50m>",
                "frequency": "<times per week>"
            }}
        ],
        "target_metrics": {{
            "stroke_rate": "<target range>",
            "stroke_length": "<target range>",
            "tempo": "<target per stroke>"
        }}
    }},
    "phase_2_development": {{
        "duration_weeks": <4-8>,
        "focus": "<progression focus>",
        "drills": [
            {{
                "name": "<drill name>",
                "purpose": "<what it develops>",
                "description": "<how to perform it>",
                "sets_reps": "<volume>",
                "frequency": "<times per week>"
            }}
        ],
        "race_simulation": "<how to practice race-pace swimming>"
    }},
    "phase_3_race_prep": {{
        "duration_weeks": <2-4>,
        "focus": "<race-specific sharpening>",
        "key_sets": [
            {{
                "name": "<set name>",
                "description": "<the set>",
                "purpose": "<race-specific benefit>"
            }}
        ],
        "mental_cues": ["<in-race thought cues to maintain technique>"]
    }},
    "expected_improvement": {{
        "time_reduction_estimate": "<realistic estimate in seconds>",
        "primary_gains_from": "<which technical changes contribute most>",
        "caveat": "<honest caveat about individual variation>"
    }}
}}"""


# ---------------------------------------------------------------------------
# Race Strategy Analysis
# ---------------------------------------------------------------------------
RACE_STRATEGY_PROMPT = """Analyze the pacing and race strategy in this swimming video.
The swimmer is racing {distance}m {stroke} in a {pool_type} pool.

Split data from CV analysis:
{split_data}

Velocity profile: {velocity_profile}

Stroke rate progression: {stroke_rate_data}

Compare this race execution against optimal pacing strategies used by elite
swimmers in this event. Consider:

1. SPLIT PATTERN: Is it negative split (second half faster), positive split
   (first half faster), or even split? How does this compare to how
   Olympic finalists typically race this event?
2. SPEED MAINTENANCE: Where does the swimmer lose the most speed? Is it
   gradual fatigue or a sudden drop?
3. STROKE RATE MANAGEMENT: Does the stroke rate increase to compensate for
   fatigue (common in sprints) or stay stable?
4. ENERGY DISTRIBUTION: Is the swimmer "dying" in the last 25m/50m? Or
   leaving time on the table by starting too conservatively?
5. TURN EFFICIENCY: How much time is gained/lost at the walls vs clean swimming?

Respond with this JSON:
{{
    "split_pattern": "<negative|positive|even>",
    "split_analysis": "<detailed breakdown of pacing>",
    "elite_pacing_comparison": {{
        "event_norm": "<how elites typically pace this event>",
        "user_vs_norm": "<how this swimmer's pacing compares>",
        "specific_example": "<e.g. 'Pan Zhanle goes out in 22.28 / comes back 24.12'>"
    }},
    "speed_loss_zones": [
        {{
            "zone": "<e.g. '35-50m'>",
            "speed_drop_pct": <percentage>,
            "likely_cause": "<fatigue|technique breakdown|turn|breathing>",
            "fix": "<recommendation>"
        }}
    ],
    "stroke_rate_analysis": {{
        "pattern": "<stable|increasing|decreasing|erratic>",
        "optimal_adjustment": "<what to change>"
    }},
    "recommended_race_plan": {{
        "target_first_split": "<time>",
        "target_second_split": "<time>",
        "key_focus_points": ["<in-race cues>"],
        "pacing_strategy": "<detailed pacing recommendation>"
    }}
}}"""


# ---------------------------------------------------------------------------
# Stroke Identification (simple, used as classifier fallback)
# ---------------------------------------------------------------------------
STROKE_IDENTIFICATION_PROMPT = """Watch this swimming video carefully and identify the
swimming stroke being performed.

The four competitive strokes are:
- Freestyle (front crawl): alternating arm rotation, flutter kick, face down
- Backstroke: alternating arm rotation, flutter kick, face up
- Breaststroke: simultaneous arm pull with glide, whip/frog kick, face down
- Butterfly: simultaneous arm recovery over water, dolphin kick, face down

Respond with ONLY this JSON (no other text):
{{
    "stroke": "<freestyle|backstroke|breaststroke|butterfly>",
    "confidence": <0.0-1.0>,
    "reasoning": "<brief explanation of what you observed>"
}}"""


# ---------------------------------------------------------------------------
# Reference Video Comparison
# ---------------------------------------------------------------------------
REFERENCE_VIDEO_COMPARISON_PROMPT = """You are given TWO swimming videos:
1. VIDEO 1 (User): The swimmer whose technique we are analyzing
2. VIDEO 2 (Reference): An elite/reference swimmer to compare against

Both swimmers are performing {stroke}. Compare them across all visible
technique elements.

Focus on OBSERVABLE differences, not assumptions. For each difference,
explain:
- What the user does vs what the reference swimmer does
- Why the reference technique is more effective (biomechanics)
- How the user can train toward the reference model

Respond with this JSON:
{{
    "comparison_summary": "<1-2 sentence overview>",
    "technique_differences": [
        {{
            "element": "<body part / phase>",
            "user_technique": "<what user does>",
            "reference_technique": "<what reference does>",
            "impact": "<how this affects speed/efficiency>",
            "drill_to_fix": "<specific drill>",
            "difficulty": "<easy|moderate|hard>"
        }}
    ],
    "what_user_does_well": [
        "<elements where user matches or exceeds reference>"
    ],
    "overall_technique_gap": "<small|moderate|large>",
    "top_priority_change": "<single most impactful change>"
}}"""
