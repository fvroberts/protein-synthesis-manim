"""A Manim walkthrough of the incompressible Navier-Stokes equations.

Render the whole video:
    manim -qh navier_stokes.py NavierStokesFull

Or a single scene, for example:
    manim -pql navier_stokes.py VortexStreet

The two simulation scenes run a real 2-D Navier-Stokes solver (fluid_solver.py).
The first render of each takes a few minutes; the result is cached in sim_cache/.
"""

from __future__ import annotations

import numpy as np
from manim import *
from scipy.special import erf

import fluid_solver

BG = "#0B0F19"
INK = "#E8ECF4"
MUTED = "#8B95A8"
WALL = "#2A3142"

C_UNSTEADY = "#FFD166"
C_CONVECT = "#F78C6B"
C_PRESSURE = "#4CC9F0"
C_VISCOUS = "#7BE495"
C_FORCE = "#C792EA"
C_MASS = "#FF6B9A"

config.background_color = BG

# Submobject indices of the terms in momentum_equation().
I_UNSTEADY, I_CONVECT, I_PRESSURE, I_VISCOUS, I_FORCE = 2, 4, 7, 9, 11
TERM_COLORS = {
    I_UNSTEADY: C_UNSTEADY,
    I_CONVECT: C_CONVECT,
    I_PRESSURE: C_PRESSURE,
    I_VISCOUS: C_VISCOUS,
    I_FORCE: C_FORCE,
}

TEX_CONVECT = r"(\mathbf{u}\cdot\nabla)\mathbf{u}"
TEX_PRESSURE = r"-\nabla p"
TEX_VISCOUS = r"\mu\nabla^{2}\mathbf{u}"
TEX_CONTINUITY = r"\nabla\cdot\mathbf{u} = 0"


# --------------------------------------------------------------------------------------
# Shared building blocks
# --------------------------------------------------------------------------------------


def momentum_equation(**kwargs) -> MathTex:
    return MathTex(
        r"\rho",
        r"\Big(",
        r"\frac{\partial \mathbf{u}}{\partial t}",
        "+",
        TEX_CONVECT,
        r"\Big)",
        "=",
        TEX_PRESSURE,
        "+",
        TEX_VISCOUS,
        "+",
        r"\mathbf{f}",
        **kwargs,
    )


def colored_equations(font_size=40) -> VGroup:
    momentum = momentum_equation(font_size=font_size)
    for i, color in TERM_COLORS.items():
        momentum[i].set_color(color)
    continuity = MathTex(TEX_CONTINUITY, font_size=font_size, color=C_MASS)
    return VGroup(momentum, continuity).arrange(DOWN, buff=0.3)


def header(title: str, tex: str | None = None, color=INK) -> VGroup:
    """Scene title pinned to the top, followed by the equation term it explains."""
    group = VGroup(Text(title, font_size=36, color=INK, weight=BOLD))
    if tex is not None:
        group.add(MathTex(tex, font_size=54, color=color))
    return group.arrange(RIGHT, buff=0.5).to_edge(UP, buff=0.45)


def caption(*lines: str, font_size=26, color=INK) -> VGroup:
    texts = [Text(line, font_size=font_size, color=color) for line in lines]
    return VGroup(*texts).arrange(DOWN, buff=0.16).to_edge(DOWN, buff=0.45)


def math_caption(before: str, tex: str, after: str = "", color=INK, font_size=26) -> VGroup:
    """A one-line caption with an inline formula."""
    parts = [Text(before, font_size=font_size, color=INK), MathTex(tex, font_size=font_size + 8, color=color)]
    if after:
        parts.append(Text(after, font_size=font_size, color=INK))
    return VGroup(*parts).arrange(RIGHT, buff=0.18).to_edge(DOWN, buff=0.5)


def fade_all(scene: Scene, run_time=1.0):
    scene.play(*[FadeOut(m) for m in scene.mobjects], run_time=run_time)


def flow_arrow(start, vec, color, scale=1.0, opacity=1.0, stroke_width=3) -> Arrow:
    start = np.asarray(start, dtype=float)
    return Arrow(
        start,
        start + scale * np.asarray(vec, dtype=float),
        buff=0,
        color=color,
        stroke_width=stroke_width,
        max_tip_length_to_length_ratio=0.3,
        max_stroke_width_to_length_ratio=8,
    ).set_opacity(opacity)


def ramp_image(values: np.ndarray, stops, colors) -> np.ndarray:
    """Map a 2-D array onto RGBA through a piecewise-linear colour ramp."""
    rgb = np.array([color_to_int_rgb(ManimColor(c)) for c in colors], dtype=float)
    out = np.empty(values.shape + (4,), dtype=np.uint8)
    for ch in range(3):
        out[..., ch] = np.interp(values, stops, rgb[:, ch])
    out[..., 3] = 255
    return out


VORTICITY_STOPS = [-1.0, -0.55, -0.2, -0.04, 0.04, 0.2, 0.55, 1.0]
VORTICITY_COLORS = ["#D6F2FF", "#3A8DFF", "#173A75", BG, BG, "#6B2010", "#FF6A2B", "#FFE9A8"]


def vorticity_legend(font_size=22) -> VGroup:
    def swatch(color):
        return Square(0.22, fill_color=color, fill_opacity=1, stroke_width=0)

    return VGroup(
        Text("vorticity (local spin):", font_size=font_size, color=MUTED),
        swatch(VORTICITY_COLORS[1]),
        Text("clockwise", font_size=font_size, color=INK),
        swatch(VORTICITY_COLORS[6]),
        Text("counter-clockwise", font_size=font_size, color=INK),
    ).arrange(RIGHT, buff=0.18)


class FieldMovie:
    """Plays back simulated vorticity frames as an ImageMobject driven by a ValueTracker."""

    def __init__(self, frames, extent, scale, width, crop=None):
        # frames: (n, ny, nx) with row index increasing in +y; extent: (x0, x1, y0, y1)
        self.frames, self.scale = frames, scale
        self.crop = crop or (slice(None), slice(None))
        self.extent = extent
        self.frame = ValueTracker(0)
        self._cached_index = None
        self.image = ImageMobject(self._rgba(0))
        self.image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bilinear"])
        self.image.width = width

    @property
    def index(self) -> int:
        return int(np.clip(round(self.frame.get_value()), 0, len(self.frames) - 1))

    def _rgba(self, i):
        w = self.frames[i][self.crop].astype(np.float32) / self.scale
        return ramp_image(np.clip(w, -1, 1), VORTICITY_STOPS, VORTICITY_COLORS)[::-1]

    def start(self):
        def update(img):
            i = self.index
            if i != self._cached_index:
                img.pixel_array = self._rgba(i)
                self._cached_index = i

        self.image.add_updater(update)

    def point(self, x, y):
        """Scene coordinates of the simulation point (x, y)."""
        x0, x1, y0, y1 = self.extent
        fx, fy = (x - x0) / (x1 - x0) - 0.5, (y - y0) / (y1 - y0) - 0.5
        return self.image.get_center() + np.array([fx * self.image.width, fy * self.image.height, 0])


# --------------------------------------------------------------------------------------
# Scenes
# --------------------------------------------------------------------------------------


class Title(Scene):
    def construct(self):
        def flow(p):
            x, y = p[0], p[1]
            return np.array([0.9 + 0.55 * np.sin(0.9 * y + 0.4 * x), 0.5 * np.sin(0.8 * x - 0.3 * y), 0.0])

        stream = StreamLines(
            flow,
            x_range=[-7.5, 7.5, 0.3],
            y_range=[-4.2, 4.2, 0.3],
            stroke_width=2,
            opacity=0.85,
            padding=1,
            colors=[C_PRESSURE, "#7B9CFF", C_FORCE],
            min_color_scheme_value=0.3,
            max_color_scheme_value=1.5,
            max_anchors_per_line=40,
        )
        title = Text("The Navier–Stokes Equations", font_size=52, weight=BOLD, color=INK)
        subtitle = Text("the laws of motion for air, water, blood and stars", font_size=28, color=MUTED)
        words = VGroup(title, subtitle).arrange(DOWN, buff=0.35)
        backdrop = BackgroundRectangle(words, color=BG, fill_opacity=0.8, buff=0.45, corner_radius=0.2)

        self.add(stream)
        stream.start_animation(warm_up=True, flow_speed=1.2, time_width=0.5)
        self.wait(0.8)
        self.play(FadeIn(backdrop), Write(title), run_time=2)
        self.play(FadeIn(subtitle, shift=0.2 * UP))
        self.wait(3)
        fade_all(self)


class EquationAnatomy(Scene):
    def construct(self):
        newton = MathTex("m", r"\mathbf{a}", "=", r"\mathbf{F}", font_size=90)
        newton_label = Text("Newton's second law", font_size=30, color=MUTED).next_to(newton, DOWN, buff=0.6)
        self.play(Write(newton), FadeIn(newton_label, shift=0.2 * UP))
        self.wait(1)

        parcel = MathTex(r"\rho", r"\frac{D\mathbf{u}}{Dt}", "=", r"\sum \frac{\mathbf{F}}{V}", font_size=80)
        parcel_label = Text("…for a tiny parcel of fluid, per unit volume", font_size=30, color=MUTED)
        parcel_label.next_to(parcel, DOWN, buff=0.6)
        self.play(
            *[ReplacementTransform(newton[i], parcel[i]) for i in range(4)],
            ReplacementTransform(newton_label, parcel_label),
            run_time=1.5,
        )
        self.wait(1.5)

        full = momentum_equation(font_size=60).shift(0.7 * UP)
        self.play(
            ReplacementTransform(parcel[0], full[0]),
            ReplacementTransform(parcel[1], full[1:6]),
            ReplacementTransform(parcel[2], full[6]),
            ReplacementTransform(parcel[3], full[7:]),
            FadeOut(parcel_label),
            run_time=2,
        )
        self.remove(*self.mobjects)
        self.add(full)

        glossary = MathTex(
            r"\rho:\ \text{density}\qquad \mathbf{u}:\ \text{velocity}\qquad"
            r" p:\ \text{pressure}\qquad \mu:\ \text{viscosity}",
            font_size=32,
            color=MUTED,
        ).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(glossary))

        lhs_brace = Brace(full[0:6], UP, color=MUTED)
        rhs_brace = Brace(full[7:], UP, color=MUTED)
        lhs_text = Text("mass × acceleration", font_size=28).next_to(lhs_brace, UP)
        rhs_text = Text("forces", font_size=28).next_to(rhs_brace, UP)
        self.play(GrowFromCenter(lhs_brace), FadeIn(lhs_text))
        self.play(GrowFromCenter(rhs_brace), FadeIn(rhs_text))
        self.wait(2)
        self.play(*[FadeOut(m) for m in (lhs_brace, rhs_brace, lhs_text, rhs_text)])

        terms = [
            (I_UNSTEADY, UP, "unsteady acceleration", "change in time at a point"),
            (I_CONVECT, DOWN, "convective acceleration", "momentum carried by the flow"),
            (I_PRESSURE, UP, "pressure gradient", "push from high to low"),
            (I_VISCOUS, DOWN, "viscosity", "internal friction"),
            (I_FORCE, UP, "body force", "gravity, magnetism…"),
        ]
        annotations = VGroup()
        for index, direction, name, meaning in terms:
            color = TERM_COLORS[index]
            brace = Brace(full[index], direction, color=color, buff=0.12)
            label = VGroup(
                Text(name, font_size=24, color=color, weight=BOLD),
                Text(meaning, font_size=20, color=MUTED),
            ).arrange(DOWN, buff=0.08)
            label.next_to(brace, direction, buff=0.12)
            self.play(full[index].animate.set_color(color), GrowFromCenter(brace), FadeIn(label), run_time=0.8)
            annotations.add(brace, label)
            self.wait(1.2)

        continuity = MathTex(TEX_CONTINUITY, font_size=60, color=C_MASS)
        continuity_label = VGroup(
            Text("incompressibility", font_size=24, color=C_MASS, weight=BOLD),
            Text("mass is neither created nor destroyed", font_size=20, color=MUTED),
        ).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        continuity_row = VGroup(continuity, continuity_label).arrange(RIGHT, buff=0.5)
        continuity_row.next_to(full, DOWN, buff=1.5)
        self.play(Write(continuity), FadeIn(continuity_label, shift=0.2 * LEFT))
        self.wait(2.5)

        summary = caption(
            "Four unknowns (three velocity components and pressure), four equations:",
            "easy to write down — notoriously hard to solve.",
        )
        self.play(FadeOut(annotations), FadeOut(continuity_label), FadeOut(glossary))
        self.play(
            VGroup(full, continuity).animate.arrange(DOWN, buff=0.5).move_to(0.4 * UP),
            FadeIn(summary, shift=0.2 * UP),
        )
        self.wait(3.5)
        fade_all(self)


class Convection(Scene):
    """Steady flow through a nozzle: no time dependence, yet parcels accelerate."""

    def construct(self):
        yc, q = -0.6, 2.2  # centre line of the channel; volume flux (sets the speeds)

        def h(x):  # channel half-width
            return 1.0 + 0.45 * (1 - np.tanh(x / 1.3))

        def dh(x):
            return -0.45 / 1.3 / np.cosh(x / 1.3) ** 2

        def velocity(x, y):
            u = q / h(x)
            return np.array([u, (y - yc) * dh(x) / h(x) * u, 0.0])

        title = header("Convective acceleration", TEX_CONVECT, C_CONVECT)
        self.play(FadeIn(title, shift=0.2 * DOWN))

        x_span = (-7.2, 7.2)
        walls = VGroup()
        for side in (1, -1):
            xs = np.linspace(*x_span, 120)
            inner = [[x, yc + side * h(x), 0] for x in xs]
            outer = [[x, yc + side * (h(x) + 0.35), 0] for x in xs[::-1]]
            wall = Polygon(*inner, *outer, fill_color=WALL, fill_opacity=1, stroke_width=0)
            edge = VMobject(stroke_color=MUTED, stroke_width=3).set_points_smoothly(inner)
            walls.add(wall, edge)

        streamlines = VGroup(
            *[
                ParametricFunction(lambda s, c=c: [s, yc + c * h(s), 0], t_range=[*x_span], stroke_width=1.5)
                .set_stroke(MUTED, opacity=0.3)
                for c in np.linspace(-0.8, 0.8, 7)
            ]
        )
        field = VGroup(
            *[
                flow_arrow([x, yc + c * h(x), 0], velocity(x, yc + c * h(x)), "#9FB3D9", 0.22, 0.55, 2.5)
                for x in np.arange(-6.3, 6.4, 0.9)
                for c in (-0.7, -0.25, 0.25, 0.7)
            ]
        )
        self.play(FadeIn(walls), Create(streamlines), run_time=1.5)
        self.play(LaggedStart(*[GrowArrow(a) for a in field], lag_ratio=0.01), run_time=1.5)

        steady = math_caption(
            "A steady flow: at every fixed point the velocity never changes,",
            r"\frac{\partial \mathbf{u}}{\partial t} = 0",
            color=C_UNSTEADY,
        )
        self.play(FadeIn(steady, shift=0.2 * UP))
        self.wait(2)

        # Parcels follow streamlines y - yc = c h(x); invert t(x) = int h/q dx for x(t).
        x_tab = np.linspace(-6.6, 5.6, 3000)
        t_tab = np.concatenate([[0], np.cumsum(h(x_tab[1:]) * np.diff(x_tab) / q)])
        clock = ValueTracker(0)

        def parcel_position(c):
            x = np.interp(clock.get_value(), t_tab, x_tab)
            return np.array([x, yc + c * h(x), 0])

        parcels, trails, arrows = VGroup(), VGroup(), VGroup()
        for c in (-0.55, 0.0, 0.55):
            dot = Dot(radius=0.11, color=C_CONVECT).set_z_index(3)
            dot.add_updater(lambda m, c=c: m.move_to(parcel_position(c)))
            glow = always_redraw(
                lambda d=dot: Dot(d.get_center(), radius=0.24, color=C_CONVECT, fill_opacity=0.25)
            )
            trails.add(TracedPath(dot.get_center, stroke_color=C_CONVECT, stroke_width=3, dissipating_time=1.2))
            arrows.add(
                always_redraw(lambda d=dot: flow_arrow(d.get_center(), velocity(*d.get_center()[:2]), INK, 0.42))
            )
            parcels.add(VGroup(glow, dot))

        middle = parcels[1][1]
        speed = DecimalNumber(1.0, num_decimal_places=2, font_size=30, color=C_CONVECT)
        speed_row = VGroup(MathTex(r"|\mathbf{u}| =", font_size=34), speed, Text("m/s", font_size=24))
        speed_row.arrange(RIGHT, buff=0.12)
        speed_label = VGroup(Text("speed of the centre parcel", font_size=20, color=MUTED), speed_row)
        speed_label.arrange(DOWN, buff=0.1)
        u_inlet = q / h(x_tab[0])

        def ride_above_wall(m):
            x = middle.get_center()[0]
            limit = config.frame_width / 2 - m.width / 2 - 0.3
            m.move_to([np.clip(x, -limit, limit), yc + h(x) + 0.95, 0])

        speed_label.add_updater(ride_above_wall)
        speed.add_updater(lambda m: m.set_value(np.linalg.norm(velocity(*middle.get_center()[:2])) / u_inlet))

        for p in parcels:
            p[1].update()
        speed_label.update()
        self.add(trails, arrows)
        self.play(FadeIn(parcels), FadeIn(speed_label))

        accelerate = math_caption(
            "…yet every parcel speeds up as it is carried into the narrow part:",
            TEX_CONVECT + r"\neq 0",
            color=C_CONVECT,
        )
        t_end = t_tab[-1]
        self.play(clock.animate.set_value(0.35 * t_end), run_time=0.35 * t_end, rate_func=linear)
        self.play(
            clock.animate.set_value(0.45 * t_end),
            FadeOut(steady, shift=0.2 * UP),
            FadeIn(accelerate, shift=0.2 * UP),
            run_time=0.1 * t_end,
            rate_func=linear,
        )
        self.play(clock.animate.set_value(t_end), run_time=0.55 * t_end, rate_func=linear)
        self.wait(0.5)

        nonlinear = caption(
            "The velocity multiplies its own gradient, so the equation is nonlinear —",
            "this is the term that makes turbulence possible.",
        )
        self.play(FadeOut(accelerate), FadeIn(nonlinear), Circumscribe(title[1], color=C_CONVECT))
        self.wait(3)
        fade_all(self)


class Pressure(Scene):
    def construct(self):
        title = header("Pressure gradient", TEX_PRESSURE, C_PRESSURE)
        self.play(FadeIn(title, shift=0.2 * DOWN))

        x0, x1, y0, y1 = -6.4, 6.4, -2.3, 2.2
        width = 2.5  # length scale of the pressure drop

        def p(x):
            return -np.tanh(x / width)

        def force(x):  # -dp/dx
            return 1 / (width * np.cosh(x / width) ** 2)

        xs = np.linspace(x0, x1, 640)
        field_img = ramp_image(np.tile(p(xs), (230, 1)), [-1, 0, 1], ["#12345F", BG, "#64202C"])
        backdrop = ImageMobject(field_img)
        backdrop.stretch_to_fit_width(x1 - x0).stretch_to_fit_height(y1 - y0).move_to([(x0 + x1) / 2, (y0 + y1) / 2, 0])
        high = Text("HIGH pressure", font_size=28, color="#FF9AA8", weight=BOLD).move_to([x0 + 1.7, y1 - 0.4, 0])
        low = Text("LOW pressure", font_size=28, color="#8FC3FF", weight=BOLD).move_to([x1 - 1.7, y1 - 0.4, 0])
        self.play(FadeIn(backdrop), FadeIn(high), FadeIn(low))

        arrows = VGroup(
            *[
                flow_arrow([x, y, 0], [force(x), 0, 0], C_PRESSURE, 2.0, 0.45, 2.5)
                for x in np.arange(-5.6, 5.7, 1.0)
                for y in np.arange(y0 + 0.5, y1 - 0.7, 0.85)
            ]
        )
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.01), run_time=1.5)
        self.play(arrows.animate.set_opacity(0.18))

        side = 1.3
        px = ValueTracker(-2.6)
        cy = (y0 + y1) / 2

        def push_length(x):
            return 0.1 + 0.5 * (p(x) + 1)

        def parcel():
            x = px.get_value()
            box = Square(side, color=INK, fill_color=INK, fill_opacity=0.12, stroke_width=3).move_to([x, cy, 0])
            left, right = x - side / 2, x + side / 2
            pushes = VGroup(
                flow_arrow([left - push_length(left), cy, 0], [push_length(left), 0, 0], "#FF9AA8"),
                flow_arrow([right + push_length(right), cy, 0], [-push_length(right), 0, 0], "#8FC3FF"),
                flow_arrow([x, cy + side / 2 + push_length(x), 0], [0, -push_length(x), 0], MUTED),
                flow_arrow([x, cy - side / 2 - push_length(x), 0], [0, push_length(x), 0], MUTED),
            )
            net = flow_arrow([x, cy - side / 2 - 1.25, 0], [4.0 * (p(left) - p(right)), 0, 0], C_PRESSURE, 1, 1, 6)
            net_label = MathTex(TEX_PRESSURE, font_size=34, color=C_PRESSURE).next_to(net, DOWN, buff=0.1)
            return VGroup(box, pushes, net, net_label)

        parcel_view = always_redraw(parcel)
        explain = caption("Pressure pushes inward on every face of a fluid parcel.")
        self.play(FadeIn(parcel_view), FadeIn(explain, shift=0.2 * UP))
        self.wait(2)

        explain2 = caption(
            "The high-pressure side pushes harder, so the net force points",
            "toward low pressure — and the parcel accelerates that way.",
        )
        self.play(FadeOut(explain), FadeIn(explain2))
        self.play(px.animate.set_value(4.6), run_time=6, rate_func=rate_functions.ease_in_sine)
        self.wait(0.5)

        explain3 = caption(
            "Uniform pressure squeezes equally from all sides and does nothing:",
            "only differences in pressure — its gradient — move the fluid.",
        )
        self.play(FadeOut(explain2), FadeIn(explain3))
        self.wait(3.5)
        fade_all(self)


class Viscosity(Scene):
    def construct(self):
        title = header("Viscous diffusion", TEX_VISCOUS, C_VISCOUS)
        self.play(FadeIn(title, shift=0.2 * DOWN))

        base_x, yc, half_width, length, nu, u_slow = -6.0, -0.55, 0.7, 5.6, 0.035, 0.12
        levels = np.linspace(-2.45, 1.35, 23)
        time = ValueTracker(1e-4)

        def profile(y):
            s = 2 * np.sqrt(nu * time.get_value())
            jet = 0.5 * (erf((half_width - (y - yc)) / s) + erf((half_width + (y - yc)) / s))
            return u_slow + (1 - u_slow) * jet

        wall = Line([base_x, levels[0] - 0.2, 0], [base_x, levels[-1] + 0.2, 0], color=MUTED, stroke_width=2)

        def arrows():
            group = VGroup()
            for y in levels:
                u = profile(y)
                if u * length > 0.05:
                    color = interpolate_color(ManimColor("#3E5C4A"), ManimColor(C_VISCOUS), min(1.0, u))
                    group.add(flow_arrow([base_x, y, 0], [u * length, 0, 0], color, stroke_width=3.5))
            return group

        def curve(stroke=C_VISCOUS):
            ys = np.linspace(levels[0], levels[-1], 300)
            return VMobject(stroke_color=stroke, stroke_width=4).set_points_as_corners(
                [[base_x + profile(y) * length, y, 0] for y in ys]
            )

        layers = always_redraw(arrows)
        tips = always_redraw(curve)
        initial = DashedVMobject(curve(MUTED), num_dashes=60).set_stroke(opacity=0.6)
        self.play(Create(wall), FadeIn(layers), Create(tips))

        intro = caption("A fast jet of fluid flowing between slower layers.")
        self.play(FadeIn(intro, shift=0.2 * UP))
        self.wait(1.5)

        laplacian = VGroup(
            MathTex(r"\nabla^2 \mathbf{u}", r"\;\propto\;", r"\overline{\mathbf{u}}_{\text{around}}", "-", r"\mathbf{u}_{\text{here}}", font_size=40),
            Text("how much a point differs from", font_size=22, color=MUTED),
            Text("the average of its neighbours", font_size=22, color=MUTED),
        ).arrange(DOWN, buff=0.15)
        laplacian[0][0].set_color(C_VISCOUS)
        laplacian.move_to([3.6, 0.6, 0])
        box = SurroundingRectangle(laplacian, color=WALL, buff=0.3, corner_radius=0.15)

        rub = caption(
            "Neighbouring layers rub against each other: fast layers drag",
            "slow ones forward, slow layers hold fast ones back.",
        )
        self.add(initial)
        self.play(FadeOut(intro), FadeIn(rub), FadeIn(box), FadeIn(laplacian))
        self.play(time.animate.set_value(6), run_time=7, rate_func=rate_functions.ease_in_out_sine)

        honey = caption(
            "Sharp differences in velocity smooth themselves out.",
            "Honey (large μ) does this quickly; air (small μ) very slowly.",
        )
        self.play(FadeOut(rub), FadeIn(honey))
        self.play(time.animate.set_value(14), run_time=4, rate_func=rate_functions.ease_in_out_sine)
        self.wait(2)
        fade_all(self)


class Incompressibility(Scene):
    def construct(self):
        title = header("Conservation of mass", TEX_CONTINUITY, C_MASS)
        self.play(FadeIn(title, shift=0.2 * DOWN))

        rate = 0.35
        panels = [
            (np.array([-3.55, -0.55, 0]), np.array([rate, rate]), r"\nabla\cdot\mathbf{u} > 0",
             "fluid would appear from nowhere", "#FF8A80"),
            (np.array([3.55, -0.55, 0]), np.array([rate, -rate]), r"\nabla\cdot\mathbf{u} = 0",
             "shape changes, area does not", C_MASS),
        ]
        time = ValueTracker(0)
        corners = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
        grid = np.array([[gx, gy] for gx in np.linspace(-0.4, 0.4, 5) for gy in np.linspace(-0.4, 0.4, 5)])

        everything = VGroup()
        for centre, diag, tex, words, color in panels:
            frame = RoundedRectangle(width=6.6, height=4.6, corner_radius=0.2, color=WALL, stroke_width=2)
            frame.move_to(centre)
            arrows = VGroup(
                *[
                    flow_arrow(centre + [dx, dy, 0], [diag[0] * dx, diag[1] * dy, 0], "#9FB3D9", 0.6, 0.45, 2)
                    for dx in np.arange(-2.8, 2.9, 0.7)
                    for dy in np.arange(-1.8, 1.3, 0.6)
                    if np.hypot(dx, dy) > 0.6
                    and abs(dx * (1 + 0.6 * diag[0])) < 3.1
                    and abs(dy * (1 + 0.6 * diag[1])) < 2.1
                ]
            )

            def warp(points, centre=centre, diag=diag):
                stretched = points * np.exp(diag * time.get_value())
                return [centre + [px, py, 0] for px, py in stretched]

            blob = always_redraw(
                lambda warp=warp, color=color: Polygon(
                    *warp(corners), color=color, fill_color=color, fill_opacity=0.25, stroke_width=3
                )
            )
            particles = always_redraw(
                lambda warp=warp: VGroup(*[Dot(p, radius=0.045, color=INK) for p in warp(grid)])
            )
            area = DecimalNumber(1.0, num_decimal_places=2, font_size=30, color=color)
            area.add_updater(lambda m, diag=diag: m.set_value(np.exp(diag.sum() * time.get_value())))
            area_row = VGroup(Text("area ×", font_size=26, color=INK), area).arrange(RIGHT, buff=0.15)
            area_row.next_to(frame, DOWN, buff=0.2)
            label = VGroup(MathTex(tex, font_size=40, color=color), Text(words, font_size=22, color=MUTED))
            label.arrange(DOWN, buff=0.1).move_to(frame.get_top() + 0.65 * DOWN)
            backing = BackgroundRectangle(label, color=BG, fill_opacity=0.85, buff=0.1)
            panel = VGroup(frame, arrows, blob, particles, backing, label, area_row)
            everything.add(panel)
            self.play(FadeIn(frame), LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.01), run_time=1.2)
            self.play(FadeIn(blob), FadeIn(particles), FadeIn(backing), FadeIn(label), FadeIn(area_row))

        self.play(time.animate.set_value(2.2), run_time=6, rate_func=rate_functions.ease_in_out_sine)
        self.wait(1)

        note = caption(
            "Liquids are almost incompressible: whatever flows into a region",
            "must flow out. Pressure adjusts instantly to keep it that way.",
            font_size=24,
        ).to_edge(DOWN, buff=0.1)
        self.play(everything.animate.shift(0.35 * UP), FadeIn(note, shift=0.2 * UP))
        self.wait(3.5)
        fade_all(self)


def _simulation_frame(title, subtitle, eq_font=28):
    heading = VGroup(
        Text(title, font_size=34, color=INK, weight=BOLD),
        Text(subtitle, font_size=24, color=MUTED),
    ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
    heading.to_corner(UL, buff=0.4)
    equations = colored_equations(font_size=eq_font).to_corner(UR, buff=0.35)
    return heading, equations


def _time_readout(times, movie):
    value = DecimalNumber(0, num_decimal_places=2, font_size=28, color=INK)
    value.add_updater(lambda m: m.set_value(times[movie.index]))
    return VGroup(MathTex("t =", font_size=32, color=MUTED), value).arrange(RIGHT, buff=0.12)


class KelvinHelmholtz(Scene):
    def construct(self):
        data = fluid_solver.kelvin_helmholtz()
        frames, times = data["vorticity"], data["times"]
        lx, ly = float(data["lx"]), float(data["ly"])

        heading, equations = _simulation_frame(
            "Kelvin–Helmholtz instability", "two streams of fluid sliding past each other"
        )
        movie = FieldMovie(frames, (0, lx, 0, ly), scale=28.0, width=10.6)
        movie.image.move_to(0.4 * DOWN)
        border = SurroundingRectangle(movie.image, color=WALL, buff=0, stroke_width=2)
        legend = vorticity_legend().next_to(movie.image, DOWN, buff=0.25).align_to(movie.image, LEFT)
        clock = _time_readout(times, movie).next_to(movie.image, DOWN, buff=0.22).align_to(movie.image, RIGHT)

        self.play(FadeIn(heading), FadeIn(equations))
        self.play(FadeIn(movie.image), Create(border), FadeIn(legend), FadeIn(clock))

        streams = VGroup()
        for y, direction in ((0.5, RIGHT), (0.1, LEFT), (0.9, LEFT)):
            for x in (0.35, 1.0, 1.65):
                start = movie.point(x, y) - 0.45 * direction
                streams.add(Arrow(start, start + 0.9 * direction, buff=0, color=INK, stroke_width=4))
        self.play(LaggedStart(*[GrowArrow(a) for a in streams], lag_ratio=0.05))
        self.wait(1.5)

        movie.start()
        self.add(movie.frame)
        n = len(frames) - 1
        self.play(FadeOut(streams), movie.frame.animate.set_value(0.04 * n), run_time=1.0, rate_func=linear)
        self.play(movie.frame.animate.set_value(n), run_time=19, rate_func=linear)
        self.wait(1)

        note = Text("Ripples grow, roll up into vortices, and the vortices merge.", font_size=22, color=INK)
        note.move_to(legend, aligned_edge=LEFT)
        self.play(FadeOut(legend, shift=0.2 * UP), FadeIn(note, shift=0.2 * UP))
        self.wait(2)
        fade_all(self)


class VortexStreet(Scene):
    def construct(self):
        data = fluid_solver.cylinder_wake()
        frames, times = data["vorticity"], data["times"]
        lx, ly = float(data["lx"]), float(data["ly"])
        h = lx / (frames.shape[2] - 1)  # vorticity lives on cell corners, 0..lx inclusive
        cx, cy, d, re = (float(data[k]) for k in ("cx", "cy", "d", "reynolds"))

        # Show the near wake, away from the outlet and the channel walls.
        i0, i1 = 0, round(13.5 / h)
        j0, j1 = round(1.0 / h), round((ly - 1.0) / h)
        crop = (slice(j0, j1 + 1), slice(i0, i1 + 1))
        x_keep, y_keep = (i0 * h, i1 * h), (j0 * h, j1 * h)

        heading, equations = _simulation_frame(
            "Flow past a cylinder", "the von Kármán vortex street"
        )
        movie = FieldMovie(frames, (*x_keep, *y_keep), scale=4.5, width=12.4, crop=crop)
        movie.image.move_to(0.45 * DOWN)
        border = SurroundingRectangle(movie.image, color=WALL, buff=0, stroke_width=2)
        scale = movie.image.width / (x_keep[1] - x_keep[0])
        cylinder = Circle(radius=scale * d / 2, color="#C9D1E0", fill_color="#C9D1E0", fill_opacity=1, stroke_width=0)
        cylinder.move_to(movie.point(cx, cy)).set_z_index(2)
        legend = vorticity_legend().next_to(movie.image, DOWN, buff=0.25).align_to(movie.image, LEFT)
        clock = _time_readout(times, movie).next_to(movie.image, DOWN, buff=0.22).align_to(movie.image, RIGHT)

        inflow = VGroup(
            *[
                Arrow(movie.point(0.25, y), movie.point(1.35, y), buff=0, color=INK, stroke_width=4)
                for y in np.linspace(y_keep[0] + 0.8, y_keep[1] - 0.8, 4)
            ]
        )
        reynolds = MathTex(
            r"\mathrm{Re} = \frac{\rho U D}{\mu} = " + f"{re:.0f}", font_size=36, color=INK
        ).move_to(movie.point(7.5, 6.0))
        reynolds_back = BackgroundRectangle(reynolds, color=BG, fill_opacity=0.8, buff=0.12)

        self.play(FadeIn(heading), FadeIn(equations))
        self.play(FadeIn(movie.image), Create(border), FadeIn(cylinder), FadeIn(legend), FadeIn(clock))
        self.play(LaggedStart(*[GrowArrow(a) for a in inflow], lag_ratio=0.1), FadeIn(reynolds_back), FadeIn(reynolds))
        self.wait(2)

        movie.start()
        self.add(movie.frame)
        n = len(frames) - 1
        self.play(
            FadeOut(inflow), FadeOut(reynolds), FadeOut(reynolds_back),
            movie.frame.animate.set_value(0.25 * n), run_time=6, rate_func=linear,
        )
        self.play(movie.frame.animate.set_value(n), run_time=18, rate_func=linear)

        note = Text(
            "Vortices shed from alternate sides, just like in the wakes of islands and chimneys.",
            font_size=22,
            color=INK,
        )
        note.next_to(movie.image, DOWN, buff=0.25)
        self.play(FadeOut(legend, shift=0.2 * UP), FadeOut(clock, shift=0.2 * UP), FadeIn(note, shift=0.2 * UP))
        self.wait(2.5)
        fade_all(self)


class Outro(Scene):
    def construct(self):
        equations = colored_equations(font_size=56).shift(1.3 * UP)
        self.play(FadeIn(equations, shift=0.2 * UP))
        self.wait(1)

        question = VGroup(
            Text("In three dimensions, nobody has proved that smooth solutions", font_size=28, color=INK),
            Text("always exist — or that they can never blow up.", font_size=28, color=INK),
        ).arrange(DOWN, buff=0.15).next_to(equations, DOWN, buff=0.9)
        prize = Text("A Millennium Prize Problem  ·  $1,000,000", font_size=36, color=C_UNSTEADY, weight=BOLD)
        prize.next_to(question, DOWN, buff=0.6)
        self.play(FadeIn(question, shift=0.2 * UP))
        self.wait(2)
        self.play(Write(prize))
        self.wait(4)
        fade_all(self, run_time=2)


SCENES = [
    Title,
    EquationAnatomy,
    Convection,
    Pressure,
    Viscosity,
    Incompressibility,
    KelvinHelmholtz,
    VortexStreet,
    Outro,
]


class NavierStokesFull(Scene):
    """Every scene above, back to back, as a single video."""

    def construct(self):
        for scene in SCENES:
            scene.construct(self)
            self.wait(0.4)
