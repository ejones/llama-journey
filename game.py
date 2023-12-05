from collections.abc import Callable
import curses
import dataclasses
from dataclasses import dataclass, field
import os
import subprocess
import pickle
import re
import sys
import textwrap
import time
import traceback
import typing
from typing import List, Optional, Tuple, Union
import random
import math


def get_grammar_term_of_type(typ):
    if typ in (int, str, float):
        return typ.__name__
    elif dataclasses.is_dataclass(typ):
        return typ.__name__
    elif typing.get_origin(typ) is not None:
        org_typ = typing.get_origin(typ)
        args = typing.get_args(typ)
        if org_typ == list:
            term = get_grammar_term_of_type(args[0])
            # TODO: move into sub rule
            return f'"[" ({term} (", " {term})*)? "]"'
        elif org_typ == tuple:
            return '"(" ' + ' ", " '.join(get_grammar_term_of_type(arg) for arg in args) + ' ")"'
        elif org_typ == typing.Union:
            return '(' + ' | '.join(get_grammar_term_of_type(arg) for arg in args) + ')'
    raise ValueError(f'Unrecognized type: {typ}')


def get_field_type_deps(typ):
    if typ in (int, str, float):
        return
    elif dataclasses.is_dataclass(typ):
        yield typ
        return
    elif typing.get_origin(typ) is not None:
        org_typ = typing.get_origin(typ)
        args = typing.get_args(typ)
        if org_typ == list:
            yield from get_field_type_deps(args[0])
            return
        elif org_typ in (tuple, typing.Union):
            for arg in args:
                yield from get_field_type_deps(arg)
            return
    raise ValueError(f'Unrecognized type: {typ}')


def get_grammar_rules_of_dataclass(cls, pos_args=None):
    rule = f'"{cls.__name__}("'

    fields = []
    optionals = []
    for field in dataclasses.fields(cls):
        if field.default is not dataclasses.MISSING or \
                field.default_factory is not dataclasses.MISSING:
            optionals.append(field)
        else:
            fields.append(field)

    for i, field in enumerate(fields):
        rule += ' '
        if i > 0:
            rule += '", " '
        if pos_args is None or pos_args <= i:
            if optionals:
                rule += f'({cls.__name__}-opts ", ")? '
            rule += f'"{field.name}=" '
        rule += get_grammar_term_of_type(field.type)

    if optionals:
        if fields:
            rule += f' (", " {cls.__name__}-opts)?'
        else:
            rule += f' {cls.__name__}-opts?'
    rule += ' ")"'

    rules = {cls.__name__: rule}

    if optionals:
        rules[f'{cls.__name__}-opt'] = ' | '.join(
            f'"{field.name}=" {get_grammar_term_of_type(field.type)}'
            for field in optionals
        )
        rules[f'{cls.__name__}-opts'] = f'{cls.__name__}-opt (", " {cls.__name__}-opt)*'

    return rules


def grammar(rule=None, pos_args=None):
    if rule is not None and pos_args is not None:
        raise ValueError('@grammar: at most one of `rule` or `pos_args` may be supplied')
    def decorator(cls):
        rules = (
            {cls.__name__: f'"{cls.__name__}(" {rule} ")"'} if isinstance(rule, str)
                else (rule or get_grammar_rules_of_dataclass(cls, pos_args=pos_args))
        )
        cls.grammar = {}
        for field in dataclasses.fields(cls):
            for dep in get_field_type_deps(field.type):
                cls.grammar.update(dep.grammar)
        cls.grammar.update(rules)
        return cls
    if isinstance(rule, type):
        cls, rule = rule, None
        return decorator(cls)
    else:
        return decorator


def format_grammar_dict(rules):
    builtins = (
        'str ::= "\\"" [0-9A-Za-z_-]* "\\""\n\n'
        'int ::= [0-9]+\n\n'
        'float ::= [0-9]+ ("." [0-9]+)?\n\n'
    )
    return builtins + '\n\n'.join(f'{k} ::= {v}' for k, v in rules.items())


def get_dataclass_deps(cls):
    yield cls
    for field in dataclasses.fields(cls):
        for dep in get_field_type_deps(field.type):
            yield from get_dataclass_deps(dep)


@dataclass
class Style:
    label: str = ""
    inset: bool = False
    border_color: Optional[int] = 7
    fill_color: Optional[int] = None
    orig_dir: str = ""


@dataclass
class Shape:
    id: str
    width: int
    depth: int
    offset: int
    in_front_of: str = ""
    style: Style = field(default_factory=Style)


@dataclass
class ShapeGroup:
    shapes: List[Shape]

    def _get_shadow_shape(self, shape):
        for s in self.shapes:
            if (s.id == shape.in_front_of if shape.in_front_of else s.in_front_of == shape.id):
                occluded = s.width > shape.width or s.width == shape.width and bool(s.in_front_of)
                return occluded, s
        return False, None

    def _get_effective_offset(self, shape):
        _, other = self._get_shadow_shape(shape)
        if other and other.offset < shape.offset:
            return max(shape.offset, other.offset + other.depth)
        return shape.offset

    def _total_width(self):
        total = 0
        for shape in self.shapes:
            occluded, other = self._get_shadow_shape(shape)
            total += 0 if occluded else shape.width
        return total

    def _non_occluded_count(self):
        total = 0
        for shape in self.shapes:
            occluded, _ = self._get_shadow_shape(shape)
            total += 0 if occluded else 1
        return total

    def get_max_depth(self):
        if not self.shapes:
            return 0
        return max(s.depth + self._get_effective_offset(s) for s in self.shapes)

    def get_min_width(self):
        return self._non_occluded_count() - 1 + self._total_width()

    def plot(self, width, *transforms):
        spacing = (width - self._total_width()) // (self._non_occluded_count() + 1)
        pos = spacing
        for s in self.shapes:
            occluded, other = self._get_shadow_shape(s)
            if other:
                if occluded:
                    continue
                yield (
                    (pos + (s.width - other.width) // 2, self._get_effective_offset(other)),
                    (other.width, other.depth),
                    other.style
                )
            yield ((pos, self._get_effective_offset(s)), (s.width, s.depth), s.style)
            pos += s.width + max(1, spacing)


def rotate(boxes):
    for (x, y), (w, h), style in boxes:
        yield (y, x), (h, w), style


def reflect_to(end_x, end_y):
    def transform(boxes):
        for (x, y), (w, h), style in boxes:
            yield (
                ((end_x - x - w if end_x else x),
                 (end_y - y - h if end_y else y)),
                (w, h),
                style
            )
    return transform


def translate(dx, dy):
    def transform(boxes):
        for (x, y), (w, h), style in boxes:
            yield (x + dx, y + dy), (w, h), style
    return transform


def transform(x, *funcs):
    for func in funcs:
        x = func(x)
    return x


def draw(boxes, size, scale=1, view_size=(0, 0), focus_pos=None):
    view_w, view_h = view_size
    width, height = size
    canvas_w = view_w or width * scale
    canvas_h = view_h or height * scale

    offsetx = (view_w - width * scale) // 2
    offsety = (view_h - height * scale) // 2
    if focus_pos:
        focusx, focusy = focus_pos
        focusx *= scale
        focusy *= scale
        offsetx = max(canvas_w // 5 - focusx, min(offsetx, canvas_w * 4 // 5 - focusx))
        offsety = max(canvas_h // 5 - focusy, min(offsety, canvas_h * 4 // 5 - focusy))

    canvas = [[0 for _ in range(canvas_w)] for _ in range(canvas_h)]

    def set_pixel(x, y, color):
        x += offsetx
        y += offsety
        if 0 <= x < canvas_w and 0 <= y < canvas_h:
            canvas[y][x] = color

    for (x, y), (w, h), style in boxes:
        x *= scale 
        y *= scale
        x_end = x + w * scale
        y_end = y + h * scale
        offset = scale - 1 if style.inset else 0
        if style.border_color is not None:
            color = style.border_color
            for i in range(x + offset, x_end - offset):
                set_pixel(i, y + offset, color)
                set_pixel(i, y_end - 1 - offset, color)
            for i in range(y + offset, y_end - offset):
                set_pixel(x + offset, i, color)
                set_pixel(x_end - 1 - offset, i, color)
        if style.fill_color is not None:
            for i in range(x + offset + 1, x_end - offset - 1):
                for j in range(y + offset + 1, y_end - offset - 1):
                    set_pixel(i, j, style.fill_color)
        if style.label:
            name = style.label[:w * scale]
            nx = x + (w * scale - len(name)) // 2
            ny = y + h * scale // 3
            for i, c in enumerate(name):
                set_pixel(nx + i, ny, c)
         
    for i in range(0, len(canvas) - 1, 2):
        for bg, fg in zip(canvas[i], canvas[i + 1]):
            c = bg if isinstance(bg, str) else fg if isinstance(fg, str) else None
            if c:
                print(f'\x1b[m{c}', end='')
            else:
                print(f'\x1b[48;5;{bg}m\x1b[38;5;{fg}m\u2584', end='')
        print('\x1b[m\r')


def plot_relative_layout(
        ahead: ShapeGroup,
        behind: ShapeGroup,
        left: ShapeGroup,
        right: ShapeGroup,
        padding=0,
        flip_sides=False):

    min_inner = 2
    inner_width = max(min_inner, ahead.get_min_width(), behind.get_min_width())
    width = inner_width + max(padding, left.get_max_depth()) + max(padding, right.get_max_depth())
    inner_height = max(min_inner, left.get_min_width(), right.get_min_width())
    height = inner_height + max(padding, ahead.get_max_depth()) + max(padding, behind.get_max_depth())
    x_shift = max(padding, left.get_max_depth())
    y_shift = max(padding, behind.get_max_depth())

    inner_box = (
        (max(padding, left.get_max_depth()), max(padding, ahead.get_max_depth())),
        (inner_width, inner_height)
    )

    def plot(side, inner_dim):
        boxes = side.plot(inner_dim)
        if flip_sides:
            boxes = transform(boxes, reflect_to(0, side.get_max_depth()))
        return boxes

    return [
        *transform(plot(ahead, inner_width), translate(x_shift, 0)),
        *transform(plot(behind, inner_width), translate(x_shift, 0), reflect_to(0, height)),
        *transform(plot(left, inner_height), rotate, translate(0, y_shift), reflect_to(0, height)),
        *transform(
            plot(right, inner_height), rotate, translate(0, y_shift), reflect_to(width, height)
        ),
    ], inner_box, (width, height)


@grammar('str ", " ("on" | "next_to") "=" str')
@dataclass
class Item:
    name: str
    on: str = ""
    next_to: str = ""
    

@grammar(pos_args=1)
@dataclass
class Feature:
    name: str
    dist_from_wall: int
    width_along_wall: int
    depth: int
    in_front_of: str = ""


@grammar(pos_args=1)
@dataclass
class WallFeature:
    name: str
    width: int

    @property
    def dist_from_wall(self):
        return 0

    @property
    def width_along_wall(self):
        return self.width

    @property
    def depth(self):
        return 0


@grammar(pos_args=1)
@dataclass
class Person:
    type: str
    at: str = ''


@grammar(pos_args=2)
@dataclass
class People:
    count: int
    type: str
    at: str = ''


@grammar(pos_args=1)
@dataclass
class Scene:
    name: str
    ahead: List[Union[Feature, WallFeature, Item, Person, People]] = field(default_factory=list)
    behind: List[Union[Feature, WallFeature, Item, Person, People]] = field(default_factory=list)
    left: List[Union[Feature, WallFeature, Item, Person, People]] = field(default_factory=list)
    right: List[Union[Feature, WallFeature, Item, Person, People]] = field(default_factory=list)

    def plot(self):
        ahead, behind, left, right = [
            ShapeGroup([
                Shape(
                   id=f.name,
                   width=f.width_along_wall,
                   depth=f.depth,
                   offset=f.dist_from_wall,
                   in_front_of=f.in_front_of,
                   style=Style(label=f.name, orig_dir=dirname)
                ) if f.depth else Shape(
                   id=f.name,
                   width=f.width_along_wall,
                   depth=1,
                   offset=-1,
                   style=Style(
                     label=f.name,
                     orig_dir=dirname,
                     border_color=(9 if 'fire' in f.name.lower() else 0),
                   )
                )
                for f in fs if isinstance(f, (Feature, WallFeature))
            ])
            for dirname, fs in [
                ('ahead', self.ahead),
                ('behind', self.behind),
                ('left', self.left),
                ('right', self.right),
            ]
        ]
        boxes, _, (width, height) = plot_relative_layout(ahead, behind, left, right, padding=1)
        return [
            ((0,0), (width + 2, height + 2), Style(inset=True, label=self.name)),
            *transform(boxes, translate(1,1)),
        ], (width + 2, height + 2)


@dataclass
class StreetFeature:
    name: str
    dist_from_street: int
    width_along_street: int
    depth: int
    desc: str = ""


class Building(StreetFeature):
    pass


class House(StreetFeature):
    pass


class Park(StreetFeature):
    pass


@dataclass
class Roundabout:
    name: str


@dataclass
class Continuation:
    name: str = ""


@grammar(pos_args=1)
@dataclass
class InventoryChange:
    items: List[Tuple[Item, float]]


@dataclass
class Equip:
    item_pos: int


@dataclass
class StartTalk:
    pass


@dataclass
class Say:
    message: str
    to: Optional[str] = None

    def as_func_repr(self):
        return f'say({repr(self.message)}, to={repr(self.to)})'


# REVIEW
@dataclass
class Hear:
    message: str
    from_: Optional[str] = None


@dataclass
class Give:
    item: Optional[Item]
    target: Optional[str]

    def as_func_repr(self):
        item_name = self.item.name if self.item else None
        return f'give({repr(item_name)}, to={repr(self.target)})'


@dataclass
class Attack:
    target: Optional[str]
    with_: Optional[Item]

    def as_func_repr(self):
        item_name = self.with_.name if self.with_ else None
        return f'attack({repr(self.target)}, with_={repr(item_name)})'


@dataclass
class Take:
    item_name: str = ""
    from_: str = ""

    def as_func_repr(self):
        item_arg = f'{repr(self.item_name)}, ' if self.item_name else ''
        return f'take({item_arg}from_={repr(self.from_)})'


@dataclass
class Go:
    target: str
    dir: str

    def as_func_repr(self):
        return f'go({repr(self.target)}, {repr(self.dir)})'


@dataclass
class Move:
    dx: int
    dy: int


@dataclass
class NewGame:
    name: str


@dataclass
class PeopleLayout:
    items: List[Tuple[Tuple[int, int], str]]


@dataclass
class UnrecognizedConsoleCommand:
    command: str


@dataclass
class StartConsole:
    pass


@dataclass
class ReplaceScene:
    scene: Scene


@grammar(pos_args=1)
@dataclass
class Inventory:
    items: List[Tuple[Item, float]]

    # TODO: get return type -> Self (or Inventory) working
    def apply_change(self, change: InventoryChange):
        orig_items = {item.name: count for item, count in self.items}
        changed = {item.name for item, _ in change.items}
        return Inventory([
            *((item, count) for item, count in self.items if item.name not in changed),
            *((item, max(0, orig_items.get(item.name, 0) + delta)) for item, delta in change.items),
        ])


@dataclass
class Street:
    name: str
    ahead: List[StreetFeature] = field(default_factory=list)
    behind: List[StreetFeature] = field(default_factory=list)
    left: List[StreetFeature] = field(default_factory=list)
    right: List[StreetFeature] = field(default_factory=list)

    def plot(self):
        ahead, behind, left, right = [
            (
                ShapeGroup([
                    Shape(
                       id=f.name,
                       width=f.width_along_street,
                       depth=f.depth,
                       offset=f.dist_from_street,
                       style=Style(label=f.name, orig_dir=dirname)
                    )
                    for f in fs
                ]) if isinstance(fs, list) else
                ShapeGroup([
                    Shape(
                         id=fs.name or type(fs).__name__,
                         width=2,
                         depth=1,
                         offset=-1,
                         style=Style(label=fs.name, border_color=0, inset=True, orig_dir=dirname)
                    )
                ])
            )
            for dirname, fs in [
                ('ahead', self.ahead),
                ('behind', self.behind),
                ('left', self.left),
                ('right', self.right),
            ]
        ]
        boxes, ((street_x, street_y), (street_w, street_h)), (width, height) = \
            plot_relative_layout(ahead, behind, left, right, flip_sides=True)

        if not isinstance(self.left, list):
            street_x -= 2
            street_w += 2
        if not isinstance(self.right, list):
            street_w += 2
        if not isinstance(self.ahead, list):
            street_y -= 2
            street_h += 2
        if not isinstance(self.behind, list):
            street_h += 2

        street_box = ((street_x, street_y), (street_w, street_h), Style(label=self.name))

        return list(transform([street_box, *boxes], translate(1,1))), (width + 2, height + 2)


def Intersection(*args, **kwargs):
    return Street(*args, **kwargs)


def get_person_box(pos, label):
    return pos, (1, 1), Style(fill_color=4, border_color=None, label=label)


def get_people_boxes(scene):
    boxes, (width, height) = scene.plot()
    directions = [
        ('ahead', scene.ahead),
        ('behind', scene.behind),
        ('left', scene.left),
        ('right', scene.right),
    ]

    for dirname, features in directions:
        if not isinstance(features, list):
            continue
        for person in features:
            if not isinstance(person, (People, Person)):
                continue
            (bx, by), (bw, bh), _ = next(
                (b for b in boxes if b[2].label == person.at and b[2].orig_dir == dirname),
                None
            ) or next(
                (b for b in boxes if b[2].orig_dir == dirname),
                boxes[0]
            )
            # REVIEW
            # FIXME - flat objects (doors)
            perimeter_points = [
                (x, y) for x, y in [
                    *((xi, by - 1) for xi in range(bx, bx + max(bw, 2) - 1)),
                    *((bx - 1, yi) for yi in range(by, by + max(bh, 2) - 1)),
                    *((bx + bw, yi) for yi in range(by, by + max(bh, 2) - 1)),
                    *((xi, by + bh) for xi in range(bx, bx + max(bw, 2) - 1)),
                ] if 0 <= x < width and 0 <= y < height
            ]
            for i in range(0, (person.count if isinstance(person, People) else 1)):
                x, y = random.choice(perimeter_points)
                yield get_person_box((x, y), person.type)


def get_init_player_pos(scene):
    _, (scenew, sceneh) = scene.plot()
    return scenew // 2, sceneh - 3


def find_nearest_box(xy, boxes):
    x, y = xy

    def distance(ox, oy):
        return math.sqrt((ox - x) ** 2 + (oy - y) ** 2)

    # REVIEW
    def distance_box(other):
        (ox, oy), (w, h), _ = other
        return min(
            distance(ox, oy),
            distance(ox + w, oy),
            distance(ox, oy + h),
            distance(ox + w, oy + h),
        )

    return min(boxes, default=None, key=distance_box)


class InferenceProcess(subprocess.Popen):
    def __init__(self, llama_main, model, prompt, grammar, eval_globals):
        super().__init__(
            [
                llama_main,
                '-m',
                model,
                '-p',
                prompt,
                '--grammar',
                grammar,
                '--in-prefix',
                ' ',
                '-r',
                '>>>',
                '-c',
                '4096',
                '-i',
                '--interactive-first',
                '--simple-io',
                '--mirostat',
                '2'
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding='utf8',
            bufsize=0
        )

        self._eval_globals = eval_globals

        self._output = ''
        while not self._output.endswith(prompt + ' '):
            chunk = self.stdout.read(1)
            self._output += chunk
            print(chunk, end='', flush=True)

        self._all_output = ''

    def _eval(self, code):
        return eval(code, {'__builtins__': {}, **self._eval_globals})

    def send_command(self, command: str):
        self.stdin.write(f'{command}\n')
        self._all_output += f'{command}\n'

        max_width = os.get_terminal_size()[0]

        self._output = ''
        first = True
        printed_width = 0
        while not self._output.endswith('>>> '):
            chunk = self.stdout.read(1)
            if first:
                first = False
                print('\r', end='', flush=True)
            self._output += chunk
            self._all_output += chunk
            if chunk == '\n':
                print('\r')
            else:
                lines = self._output.splitlines()
                if lines[-1] and (
                        lines[-1].startswith('Scene(') or
                        lines[-1].startswith('Street(') or
                        lines[-1].startswith('Inventory')
                    ):
                    loader = [' '] * 20
                    loader_start = len(self._output) % len(loader)
                    for i in range(len(loader) // 2 + 1):
                        loader[(loader_start + i) % len(loader)] = '='
                    print('[', *loader, ']', ' ' * 10, sep='', end='\r', flush=True)
                else:
                    printed_width += len(chunk)
                    if printed_width > max_width:
                        print('\r' + ' ' * (max_width - 1), end='\r')
                        printed_width = len(chunk)
                    print(chunk, end='', flush=True)
        print('\r           \r', end='', flush=True)

    def get_last_messages(self):
        for line in self._output.splitlines()[:-1]:
            try:
                obj = self._eval(line)
            except SyntaxError:
                obj = None
                yield line
            if isinstance(obj, str):
                yield line

    def get_last_objects(self):
        lines = self._output.splitlines()
        obj = None
        for i in range(len(lines) - 1, -1, -1):
            try:
                obj = self._eval(lines[i])
            except SyntaxError:
                continue
            if not isinstance(obj, str):
                yield obj

    def get_last_object(self, *default_arg):
        return next(self.get_last_objects(), *default_arg)


class GameState:
    def __init__(self, scale, scrollback):
        self.log = []
        self.scale = scale
        self.scrollback = scrollback
        self._reset()

    def _reset(self):
        self.hitbox = None
        self.nearest_person = ''
        self.equipped_item = 0
        self.messages = []
        self.scene = None
        self.scene_desc = None
        self.inventory = None
        self.people_boxes = None
        self.posx, self.posy = None, None
        self.nearest_person = None

    def _set_scene(self, scene, desc):
        self.scene = scene
        self.scene_desc = desc
        try:
            self.posx, self.posy = get_init_player_pos(scene)
            self.people_boxes = list(get_people_boxes(scene))
        except Exception as exc:
            raise RuntimeError(f'Bad scene {repr(scene)}') from exc
        self.log.append(
            ('', PeopleLayout([(xy, style.label) for xy, _, style in self.people_boxes]))
        )

    def _move(self, move: Move):
        boxes, _ = self.scene.plot()
        self.posx += move.dx
        self.posy += move.dy
        self.hitbox = next(
            (style
             for (bx, by), (w, h), style in reversed(boxes)
             if bx <= self.posx < (bx + w) and by <= self.posy < (by + h)),
            ''
        )
        nearest_person_box = find_nearest_box((self.posx, self.posy), self.people_boxes)
        self.nearest_person = nearest_person_box[2].label if nearest_person_box else ''

    def _msg(self, message: str, wrap=False):
        if message:
            self.messages.append((message, wrap))

    def get_equipped(self):
        if self.equipped_item:
            return self.inventory.items[self.equipped_item - 1][0]
        else:
            return None
        
    def handle(self, command, comment='') -> bool:
        self.log.append((comment, command))

        if command is None:
            self._msg(comment)
            return True
        elif isinstance(command, Move):
            self._move(command)
            return True
        elif isinstance(command, StartTalk):
            if self.nearest_person:
                return True
            else:
                self._msg('No one is nearby!')
                return False
        elif isinstance(command, Say):
            self._msg(repr(command.message))
            return True
        elif isinstance(command, Hear):
            self._msg(command.message, wrap=True)
            return True
        elif isinstance(command, Equip):
            if command.item_pos > len(self.inventory.items):
                self._msg(f'No item {command.item_pos}')
                self.equipped_item = 0
                return False
            else:
                self.equipped_item = command.item_pos
                self._msg(f'You equipped {self.get_equipped().name}')
                return True
        elif isinstance(command, Give):
            # TODO validate item & target
            if not command.item:
                # REVIEW
                self._msg('No item equipped!')
                return False
            if not command.target:
                self._msg('No one is nearby!')
                return False
            self._msg(f'You give {command.item.name} to {command.target}')
            return True
        elif isinstance(command, Attack):
            # TODO validate item & target
            if not command.with_:
                # REVIEW
                self._msg('No item equipped!')
                return False
            if not command.target:
                self._msg('No one is nearby!')
                return False
            self._msg(f'You attack {command.target} with {command.with_.name}')
            return True
        elif isinstance(command, Inventory):
            self.inventory = command
            return True
        elif isinstance(command, InventoryChange):
            self.inventory = self.inventory.apply_change(command)
            if comment:
                self._msg(comment)
            self._msg(
                'You get ' + ', '.join((f'{count} {item.name}' for item, count in command.items))
            )
            return True
        elif isinstance(command, (Take, Go)):
            return True
        elif isinstance(command, Scene):
            self._set_scene(command, comment)
            self._msg(comment)
            return True
        elif isinstance(command, ReplaceScene):
            self._set_scene(command.scene, comment)
            return True
        elif isinstance(command, NewGame):
            self._reset()
            return True
        elif isinstance(command, UnrecognizedConsoleCommand):
            self._msg('Unrecognized command')
            return False # REVIEW
        elif isinstance(command, PeopleLayout):
            self.people_boxes = [get_person_box(pos, label) for pos, label in command.items]
            return True

        raise ValueError(f'Unrecognized command {repr(command)}')

    def save(self, file):
        pickle.dump(self.log, file)


def curses_prompt(stdscr, prompt):
    print(prompt, end='', flush=True)
    curses.echo()
    try:
        return stdscr.getstr().decode('utf8')
    finally:
        curses.noecho()


def draw_game(state: GameState):
    print("\x1b[2J\x1b[1;1H", end="")
    if not state.scene or not state.inventory:
        return # REVIEW
    boxes, size = state.scene.plot()
    player_box = (
        (state.posx, state.posy), (1, 1), Style(fill_color=8, label='you', border_color=None)
    )
    term_w, term_h = os.get_terminal_size()
    draw(
        [*boxes, *state.people_boxes, player_box],
        size,
        scale=state.scale,
        view_size=(term_w, 2 * (term_h - 10 - state.scrollback)),
        focus_pos=(state.posx, state.posy)
    )
    screen_width = os.get_terminal_size()[0]
    inv_message = ', '.join((f'{i.name} ({cnt})' for (i, cnt) in state.inventory.items))
    print('\r')
    print('arrow keys: move | t: talk | 1-9: equip\r')
    print('g: give | a: attack | enter: go into OR pick up items\r')
    print('~: console - `newperson <name> <desc>`, `newdoor <name> <desc>`, `newgame <name>`\r')
    print('\r')
    print(('You have ' + inv_message)[:screen_width], end='\r\n')
    print(state.hitbox.label if state.hitbox else '', '/', state.nearest_person, '\r')
    print('-' * screen_width, end='\r\n')
    displayable_messages = [
        chunk
        for msg, wrap in state.messages
        for chunk in (textwrap.wrap(msg, screen_width) if wrap else [msg])
    ]
    for chunk in displayable_messages[-state.scrollback:]:
        print(chunk[:screen_width], end='\r\n')


def replay_typing(msg: str, chunk_size: int, delay: Union[float, Callable]):
    for i in range(0, len(msg), chunk_size):
        print(msg[i:i + chunk_size], end='', flush=True)
        time.sleep(delay() if isinstance(delay, Callable) else delay)


# TODO: take a config (init params for state) rather than a state
def replay_game(state: GameState, command_log: List[Tuple[str, object]]):
    for i in range(len(command_log)):
        msg, cmd = command_log[i]
        # TODO better
        if isinstance(cmd, Scene):
            i += 1
            while i < len(command_log) and not isinstance(command_log[i][1], (Scene, ReplaceScene)):
                i += 1
            if i < len(command_log) and isinstance(command_log[i][1], ReplaceScene):
                msg, cmd = command_log[i]
                cmd = cmd.scene
        elif isinstance(cmd, ReplaceScene):
            continue
        if isinstance(cmd, NewGame):
            replay_typing(f'> newgame {cmd.name}', 1, lambda: random.randint(20, 200) / 1000)
        if msg:
            replay_typing(msg, 5, 0.085)
        elif isinstance(cmd, Say):
            print(f'say to {cmd.to}:', end=' ', flush=True)
            replay_typing(cmd.message, 1, lambda: random.randint(20, 200) / 1000)
        elif isinstance(cmd, Hear):
            replay_typing(cmd.message, 5, 0.085)
        state.handle(cmd, msg)
        time.sleep(0.2)
        draw_game(state)
        if isinstance(cmd, (Give, InventoryChange, Attack)):
            time.sleep(0.8)
    time.sleep(2)


def restore_prompt(command_log: List[Tuple[str, object]]):
    result = ''
    # HACK FIXME - first 3 logs are Scene, People, and Inventory
    for i in range(3, len(command_log)):
        msg, cmd = command_log[i]
        # TODO better
        if isinstance(cmd, Scene):
            i += 1
            while i < len(command_log) and not isinstance(command_log[i][1], (Scene, ReplaceScene)):
                i += 1
            if i < len(command_log) and isinstance(command_log[i][1], ReplaceScene):
                msg, cmd = command_log[i]
                cmd = cmd.scene
        if hasattr(cmd, 'as_func_repr'):
            result += f'{cmd.as_func_repr()}\n'
        elif isinstance(cmd, Hear):
            result += msg + ('\n' if msg else '') + cmd.message + '\n>>> '
        elif isinstance(cmd, (Scene, Inventory, InventoryChange)) or cmd is None:
            # TODO somehow declare commands that the game runs
            result += (
                msg + ('\n' if msg and cmd is not None else '') +
                (repr(cmd) if cmd is not None else '') + '\n>>> '
            )
    return result


def curses_main(stdscr, inference_proc):
    stdscr.refresh()
    state = GameState(scale=6, scrollback=6)

    scene = inference_proc.get_last_object()
    state.handle(scene)

    inventory = None
    for obj in reversed(list(inference_proc.get_last_objects())):
        if isinstance(obj, Inventory):
            inventory = obj
        elif isinstance(obj, InventoryChange) and inventory:
            inventory = inventory.apply_change(obj)
    if inventory is None:
        raise RuntimeError("no inventory found/generated!")
    state.handle(inventory)

    if len(sys.argv) > 4:
        log = pickle.load(open(sys.argv[4], 'rb'))
        replay_game(state, log)
        exit()

    draw_game(state)
    
    while True:
        key = stdscr.getch()

        if key == curses.KEY_UP:
            state.handle(Move(0, -1))
        elif key == curses.KEY_DOWN:
            state.handle(Move(0, 1))
        elif key == curses.KEY_RIGHT:
            state.handle(Move(1, 0))
        elif key == curses.KEY_LEFT:
            state.handle(Move(-1, 0))
        elif key == ord('t'):
            if state.handle(StartTalk()):
                msg = curses_prompt(stdscr, f'Say to {state.nearest_person}: ')
                command = Say(msg, to=state.nearest_person)
                state.handle(command)
                inference_proc.send_command(command.as_func_repr())
                # TODO: fix state changes from iteractions
                state.handle(Hear('\n'.join(inference_proc.get_last_messages())))
                obj = inference_proc.get_last_object(None)
                if obj:
                    state.handle(obj)
        elif ord('1') <= key <= ord('9'):
            state.handle(Equip(int(chr(key))))
        elif key == ord('g'):
            command = Give(state.get_equipped(), state.nearest_person)
            if state.handle(command):
                inference_proc.send_command(command.as_func_repr())
                obj = inference_proc.get_last_object(None)
                if not isinstance(obj, InventoryChange):
                    obj = InventoryChange([(command.item, -1)])
                state.handle(obj, '\n'.join(inference_proc.get_last_messages()))
        elif key == ord('a'):
            # TODO
            command = Attack(state.nearest_person, with_=state.get_equipped())
            if state.handle(command):
                inference_proc.send_command(command.as_func_repr())
                state.handle(None, '\n'.join(inference_proc.get_last_messages())) # TODO
                obj = inference_proc.get_last_object(None)
                if obj:
                    state.handle(obj)
        elif key == 10 or key == 13:
            # REVIEW
            if state.hitbox.label == state.scene.name:
                _, _, near_box = find_nearest_box(
                    (state.posx, state.posy), state.scene.plot()[0]
                )
                command = Take(from_=near_box.label)
                state.handle(command)
                inference_proc.send_command(command.as_func_repr())
                state.handle(inference_proc.get_last_object(None), '\n'.join(inference_proc.get_last_messages()))
            else:
                command = Go(state.hitbox.label, state.hitbox.orig_dir)
                state.handle(command)
                inference_proc.send_command(command.as_func_repr())
                state.handle(inference_proc.get_last_object(), '\n'.join(inference_proc.get_last_messages()))
        elif key == ord('`'):
            cmd = curses_prompt(stdscr, '> ')
            if cmd.startswith('newgame '):
                game_name = cmd[8:]
                state.handle(NewGame(game_name))

                stdscr.clear()
                print('\r')

                inference_proc.send_command(f'start_game({repr(game_name)})')
                obj = inference_proc.get_last_object()
                assert isinstance(obj, (Scene, Street)), f'Expected Scene, got {repr(obj)}'
                state.handle(obj)

                inference_proc.send_command('inventory')
                obj = inference_proc.get_last_object()
                assert isinstance(obj, Inventory), f'Expected Inventory, got {repr(obj)}'
                state.handle(obj)
            elif cmd.startswith('newperson ') or cmd.startswith('newdoor '):
                cmdtype, name, desc = cmd.split(None, 2)
                newobj = Person(name) if cmdtype == 'newperson' else WallFeature(name, 1)
                state.handle(
                    ReplaceScene(
                        Scene(
                            name=state.scene.name,
                            # TODO determine most appropriate direction
                            ahead=[*state.scene.ahead, newobj],
                            behind=state.scene.behind,
                            left=state.scene.left,
                            right=state.scene.right,
                        )
                    ),
                    f'{state.scene_desc} Ahead is also {desc}.'
                )
            else:
                state.handle(UnrecognizedConsoleCommand(cmd))
        elif key == ord('q'):
            with open('game.bin', 'wb') as f:
                state.save(f)
            return

        # Go(...) if walk out of scene
        if key in (curses.KEY_UP, curses.KEY_RIGHT, curses.KEY_DOWN, curses.KEY_LEFT) \
                and not state.hitbox:
            _, _, near_box = find_nearest_box((state.posx, state.posy), state.scene.plot()[0])
            command = Go(near_box.label, near_box.orig_dir)
            state.handle(command)
            print('loading...', end='\r', flush=True)
            inference_proc.send_command(command.as_func_repr())
            state.handle(inference_proc.get_last_object(), '\n'.join(inference_proc.get_last_messages()))

        draw_game(state)


def main():
    if len(sys.argv) < 3:
        print('usage: python3 scene.py <llama-cpp-main-path> <model-path> <prompt-file>')
        exit(1)

    grammar_rules = {
        'root': '(Inventory | scene-cmd | inv-cmd | dialogue) "\\n>>>   "',
        'scene-cmd': '"You " [^\\n]+ "\\n" Scene',

        # REVIEW: is this enough?
        'dialogue': '"\\"" [ !#-\\[\\]-z]* "\\"" ("\\n" inv-cmd | scene-cmd)?',

        # REVIEW: prevent InventoryChange interpreted as message line
        'inv-cmd': '([0-9A-HJ-Za-z] [^\\n]+ "\\n")? InventoryChange',

        **Scene.grammar,
        **Inventory.grammar,
        **InventoryChange.grammar,
    }

    inference_proc = InferenceProcess(
        llama_main=sys.argv[1],
        model=sys.argv[2],
        prompt=open(sys.argv[3]).read(),
        grammar=format_grammar_dict(grammar_rules),
        eval_globals={
            dep.__name__: dep
            for cls in (Scene, Inventory, InventoryChange)
            for dep in get_dataclass_deps(cls)
        },
    )

    curses.wrapper(lambda stdscr: curses_main(stdscr, inference_proc))


if __name__ == '__main__':
    main()
