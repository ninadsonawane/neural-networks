import { Component, Suspense, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Canvas, useFrame, useThree, type ThreeEvent } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { AdditiveBlending, BackSide, Box3, BufferAttribute, BufferGeometry, CanvasTexture, Group, Line, LineBasicMaterial, MathUtils, Mesh, MeshBasicMaterial, MeshPhysicalMaterial, Object3D, PerspectiveCamera, SphereGeometry, SRGBColorSpace, TorusGeometry, Vector3 } from 'three';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import type { ExplorerModel, ExplorerPart, PartCategory, Vec3 } from '../explorer/types';
import './MuseumScene.css';

export interface MuseumSceneProps {
  model: ExplorerModel;
  explosion: number;
  selectedId: string | null;
  isolated: boolean;
  visibleCategories: PartCategory[];
  mode: 'learn' | 'experiment';
  playing: boolean;
  hasRun?: boolean;
  step: number;
  resetKey: number;
  reducedMotion: boolean;
  onSelect: (id: string | null) => void;
  onHover?: (id: string | null) => void;
  flat: boolean;
}

const MINT = '#a7f5d8';
const AMBER = '#d8a573';
const CATEGORY_BASE: Record<PartCategory, string> = {
  tokens: '#8e9fae', embeddings: '#3f6b78', memory: '#426e69', gates: '#85745b',
  attention: '#416a87', residual: '#625c88', output: '#7cae9e',
};
const positionFor = (part: ExplorerPart, explosion: number): Vec3 => part.position.map((v, i) => MathUtils.lerp(v, part.explodedPosition[i], explosion)) as Vec3;
const related = (part: ExplorerPart, selectedId: string | null, parts: ExplorerPart[]) => part.id === selectedId || part.parentId === selectedId || parts.find(p => p.id === selectedId)?.parentId === part.id;
function displayedParts(props: MuseumSceneProps) {
  return props.model.parts.filter(part => props.visibleCategories.includes(part.category)
    && (!part.detailOnly || (props.explosion > .08 && related(part, props.selectedId, props.model.parts)))
    && (!props.isolated || related(part, props.selectedId, props.model.parts)));
}

function makeLabel(text: string, foreground = '#ecfff9') {
  const canvas = document.createElement('canvas');
  canvas.width = 256; canvas.height = 128;
  const context = canvas.getContext('2d');
  if (context) {
    context.clearRect(0, 0, 256, 128);
    context.font = '600 48px "DM Sans Variable", Arial, sans-serif';
    context.textAlign = 'center'; context.textBaseline = 'middle'; context.fillStyle = foreground;
    context.fillText(text.length > 7 ? `${text.slice(0, 6)}…` : text, 128, 66, 235);
  }
  const texture = new CanvasTexture(canvas);
  texture.colorSpace = SRGBColorSpace;
  return texture;
}

type SceneResources = ReturnType<typeof makeResources>;
function makeResources() {
  return {
    box: new RoundedBoxGeometry(1, 1, 1, 3, .1),
    sphere: new SphereGeometry(.5, 24, 16),
    bead: new SphereGeometry(.5, 12, 8),
    ring: new TorusGeometry(.38, .085, 12, 48),
    fineRing: new TorusGeometry(.5, .012, 6, 64),
    pinMaterial: new MeshPhysicalMaterial({ color: '#adc5d0', metalness: .72, roughness: .3 }),
    cellMaterial: new MeshPhysicalMaterial({ color: '#294957', metalness: .45, roughness: .35, emissive: '#35685f', emissiveIntensity: .25 }),
    signalMaterial: new MeshBasicMaterial({ color: MINT, toneMapped: false }),
  };
}

function TokenFace({ part, dimmed }: { part: ExplorerPart; dimmed: boolean }) {
  const texture = useMemo(() => makeLabel(part.label, part.ghost ? AMBER : '#effff9'), [part.label, part.ghost]);
  useEffect(() => () => texture.dispose(), [texture]);
  return <mesh position={[0, 0, .506]}>
    <planeGeometry args={[.88, .68]} />
    <meshBasicMaterial map={texture} transparent opacity={dimmed ? .2 : .95} depthWrite={false} toneMapped={false} />
  </mesh>;
}

function InnerLattice({ resources, kind }: { resources: SceneResources; kind: ExplorerPart['kind'] }) {
  const instance = useRef<import('three').InstancedMesh>(null);
  const count = kind === 'bundle' ? 15 : 9;
  useEffect(() => {
    if (!instance.current) return;
    const object = new Object3D();
    for (let i = 0; i < count; i++) {
      const cols = kind === 'bundle' ? 5 : 3;
      object.position.set((i % cols - (cols - 1) / 2) * .145, (Math.floor(i / cols) - 1) * .17, .502);
      object.scale.set(.055, .055, .07); object.updateMatrix();
      instance.current.setMatrixAt(i, object.matrix);
    }
    instance.current.instanceMatrix.needsUpdate = true;
  }, [count, kind]);
  return <instancedMesh ref={instance} args={[resources.bead, resources.cellMaterial, count]} />;
}

function Part({ part, resources, registry, props, hoveredId, canClick }: {
  part: ExplorerPart; resources: SceneResources; registry: Map<string, Group>;
  props: MuseumSceneProps; hoveredId: string | null; canClick: () => boolean;
}) {
  const group = useRef<Group>(null);
  const shell = useRef<Mesh>(null);
  const initialPosition = useRef(positionFor(part, props.explosion));
  const selected = related(part, props.selectedId, props.model.parts);
  const muted = props.selectedId !== null && !selected;
  const hovered = hoveredId === part.id;
  const color = part.ghost ? AMBER : (part.color || CATEGORY_BASE[part.category]);
  const material = useMemo(() => new MeshPhysicalMaterial({
    color, metalness: .4, roughness: .29, clearcoat: .7, clearcoatRoughness: .3,
    transparent: true, opacity: 0, depthWrite: true,
    emissive: part.ghost ? AMBER : MINT, emissiveIntensity: 0,
  }), [color, part.ghost]);
  useEffect(() => () => material.dispose(), [material]);
  useEffect(() => {
    if (group.current) registry.set(part.id, group.current);
    return () => { registry.delete(part.id); };
  }, [part.id, registry]);
  useFrame((state, delta) => {
    if (!group.current) return;
    const target = positionFor(part, props.explosion);
    const rate = props.reducedMotion ? 1 : 1 - Math.exp(-8 * Math.min(delta, .05));
    group.current.position.x = MathUtils.lerp(group.current.position.x, target[0], rate);
    group.current.position.y = MathUtils.lerp(group.current.position.y, target[1], rate);
    group.current.position.z = MathUtils.lerp(group.current.position.z, target[2], rate);
    const opacity = muted ? .16 : part.ghost ? .22 : .97;
    material.opacity = MathUtils.lerp(material.opacity, opacity, rate);
    material.depthWrite = opacity > .4;
    const pulse = props.playing && part.active ? (1 + Math.sin(state.clock.elapsedTime * 3.5)) * .045 : 0;
    const emissive = muted ? 0 : selected ? .18 : hovered ? .1 : part.active ? .045 + pulse : 0;
    material.emissiveIntensity = MathUtils.lerp(material.emissiveIntensity, emissive, rate);
    if (Math.abs(group.current.position.x - target[0]) + Math.abs(group.current.position.y - target[1]) + Math.abs(group.current.position.z - target[2]) > .001
      || Math.abs(material.opacity - opacity) > .001 || Math.abs(material.emissiveIntensity - emissive) > .001) state.invalidate();
  });
  const select = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation();
    if (event.delta <= 6 && canClick()) props.onSelect(part.id);
  };
  const round = part.kind === 'sphere';
  const valve = part.kind === 'valve';
  return <group ref={group} position={initialPosition.current}>
    <group scale={part.scale} onClick={select}
      onPointerOver={event => { event.stopPropagation(); props.onHover?.(part.id); }}
      onPointerOut={event => { event.stopPropagation(); props.onHover?.(null); }}>
      <mesh ref={shell} geometry={round ? resources.sphere : valve || part.kind === 'ring' ? resources.ring : resources.box} material={material} castShadow receiveShadow />
      {part.kind === 'card' && <TokenFace part={part} dimmed={muted} />}
      {(part.kind === 'slab' || part.kind === 'bundle' || part.kind === 'head') && !muted && <InnerLattice resources={resources} kind={part.kind} />}
      {valve && <>
        <mesh geometry={resources.box} material={material} scale={[.65, .085, .2]} rotation={[0, 0, Math.PI * (.12 + (part.value ?? .5) * .72)]} />
        <mesh geometry={resources.bead} material={resources.pinMaterial} scale={[.14, .14, .16]} position={[0, 0, .09]} />
      </>}
      {round && !muted && <mesh geometry={resources.fineRing} material={resources.pinMaterial} rotation={[.2, .4, -.35]} scale={1.1} />}
      {!part.ghost && part.kind !== 'ring' && !round && !valve && !muted && <mesh geometry={resources.box} material={resources.signalMaterial} position={[0, -.32, .507]} scale={[selected ? .53 : .17, .018, .016]} />}
      {selected && !round && !valve && <mesh geometry={resources.box} scale={1.035}>
        <meshBasicMaterial color={MINT} side={BackSide} transparent opacity={.22} depthWrite={false} />
      </mesh>}
    </group>
  </group>;
}

function Strand({ link, registry, props }: { link: ExplorerModel['links'][number]; registry: Map<string, Group>; props: MuseumSceneProps }) {
  const count = 25;
  const line = useMemo(() => {
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new BufferAttribute(new Float32Array(count * 3), 3));
    const material = new LineBasicMaterial({ color: link.kind === 'gradient' ? AMBER : link.kind === 'residual' ? '#a2a2ce' : MINT, transparent: true, opacity: .16, depthWrite: false });
    const object = new Line(geometry, material); object.frustumCulled = false;
    return object;
  }, [link.kind]);
  useEffect(() => () => { line.geometry.dispose(); line.material.dispose(); }, [line]);
  useFrame((state) => {
    const from = registry.get(link.from), to = registry.get(link.to);
    line.visible = !!from && !!to;
    if (!from || !to) return;
    const positions = line.geometry.attributes.position as BufferAttribute;
    const weight = MathUtils.clamp(Math.abs(link.weight ?? .45), 0, 1);
    const strength = link.kind === 'attention' ? .07 + weight * .5 : .1 + props.explosion * .16;
    const selectedLink = props.selectedId === link.from || props.selectedId === link.to;
    const pulse = props.playing && (link.kind === 'attention' || link.kind === 'flow' || link.kind === 'memory') ? .75 + .25 * Math.sin(state.clock.elapsedTime * 3 - weight * 6) : 1;
    line.material.opacity = props.selectedId ? selectedLink ? .8 : .04 : strength * pulse;
    const rise = link.kind === 'attention' ? .25 + weight * .4 : link.kind === 'residual' ? -.26 : .06;
    for (let i = 0; i < count; i++) {
      const t = i / (count - 1);
      positions.setXYZ(i, MathUtils.lerp(from.position.x, to.position.x, t), MathUtils.lerp(from.position.y, to.position.y, t), MathUtils.lerp(from.position.z, to.position.z, t) + Math.sin(t * Math.PI) * rise);
    }
    positions.needsUpdate = true;
  });
  return <primitive object={line} />;
}

function Traveller({ props, resources, registry }: { props: MuseumSceneProps; resources: SceneResources; registry: Map<string, Group> }) {
  const traveller = useRef<Group>(null);
  const orbit = useRef<Mesh>(null);
  const phase = useRef(0);
  const previousStep = useRef(props.step);
  const previousRun = useRef(props.hasRun ?? props.playing);
  const previousReset = useRef(props.resetKey);
  const previousArchitecture = useRef(props.model.architecture);
  const token = props.model.parts.find(part => part.tokenIndex === props.step && part.kind === 'card');
  const texture = useMemo(() => makeLabel(token?.label ?? '•'), [token?.label]);
  useEffect(() => () => texture.dispose(), [texture]);
  useFrame((_, delta) => {
    if (!traveller.current) return;
    const hasRun = props.hasRun ?? props.playing;
    traveller.current.visible = props.mode === 'experiment' && hasRun;
    if (!hasRun || previousStep.current !== props.step || (hasRun && !previousRun.current) || previousReset.current !== props.resetKey || previousArchitecture.current !== props.model.architecture) phase.current = 0;
    previousStep.current = props.step; previousRun.current = hasRun; previousReset.current = props.resetKey;
    previousArchitecture.current = props.model.architecture;
    if (props.playing) phase.current = Math.min(phase.current + Math.min(delta, .05) / .28, .999);
    const entry = token ? props.model.flow.indexOf(token.id) : 0;
    const route = props.model.flow.slice(Math.max(0, entry)).map(id => registry.get(id)).filter((node): node is Group => !!node);
    if (route.length < 2) { traveller.current.visible = false; return; }
    const path = phase.current * (route.length - 1), i = Math.floor(path), t = path - i;
    const from = route[i].position, to = route[Math.min(i + 1, route.length - 1)].position;
    traveller.current.position.lerpVectors(from, to, t);
    traveller.current.position.z += .55 + Math.sin(t * Math.PI) * .13;
    if (orbit.current && props.playing) orbit.current.rotation.z += Math.min(delta, .05) * .65;
  });
  return <group ref={traveller} visible={false}>
    <mesh geometry={resources.sphere} material={resources.signalMaterial} scale={.15} />
    <mesh ref={orbit} geometry={resources.fineRing} scale={.44}>
      <meshBasicMaterial color={MINT} transparent opacity={.65} blending={AdditiveBlending} depthWrite={false} />
    </mesh>
    <mesh geometry={resources.box} position={[.18, .18, .1]} scale={[.36, .25, .055]}>
      <meshPhysicalMaterial color="#679c92" metalness={.2} roughness={.25} emissive={MINT} emissiveIntensity={.12} />
    </mesh>
    <mesh position={[.18, .18, .132]}>
      <planeGeometry args={[.33, .2]} /><meshBasicMaterial map={texture} transparent toneMapped={false} />
    </mesh>
  </group>;
}

function CameraRig({ props, parts }: { props: MuseumSceneProps; parts: ExplorerPart[] }) {
  const { camera, size, invalidate } = useThree();
  const controls = useRef<OrbitControlsImpl>(null);
  const target = useRef<{ center: Vector3; position: Vector3 } | null>(null);
  const previousReset = useRef(props.resetKey);
  const boundsKey = JSON.stringify(parts.map(part => [part.id, positionFor(part, props.explosion), part.scale]));
  useEffect(() => {
    const orbit = controls.current;
    if (!orbit || !(camera instanceof PerspectiveCamera)) return;
    const box = new Box3();
    const vectors = JSON.parse(boundsKey) as [string, Vec3, Vec3][];
    for (const [, p, scale] of vectors) {
      box.expandByPoint(new Vector3(p[0] - scale[0] / 2, p[1] - scale[1] / 2, p[2] - scale[2] / 2));
      box.expandByPoint(new Vector3(p[0] + scale[0] / 2, p[1] + scale[1] / 2, p[2] + scale[2] / 2));
    }
    if (box.isEmpty()) box.set(new Vector3(-2, -2, -1), new Vector3(2, 2, 1));
    const center = box.getCenter(new Vector3());
    const isReset = previousReset.current !== props.resetKey;
    previousReset.current = props.resetKey;
    const direction = isReset ? new Vector3(5.8, 3.5, 14).normalize() : camera.position.clone().sub(orbit.target).normalize();
    const right = new Vector3().crossVectors(camera.up, direction).normalize();
    const up = new Vector3().crossVectors(direction, right).normalize();
    let width = 0, height = 0, depth = 0;
    for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y]) for (const z of [box.min.z, box.max.z]) {
      const v = new Vector3(x, y, z).sub(center);
      width = Math.max(width, Math.abs(v.dot(right))); height = Math.max(height, Math.abs(v.dot(up))); depth = Math.max(depth, Math.abs(v.dot(direction)));
    }
    const aspect = size.width / Math.max(size.height, 1);
    const safeWidth = size.width < 760 ? .84 : props.selectedId ? .69 : .86;
    const safeHeight = size.width < 760 ? .78 : .85;
    const distance = Math.max(3.1, Math.max(width / (aspect * safeWidth), height / safeHeight) / Math.tan(camera.fov * Math.PI / 360) + depth);
    // The app insets the canvas around its dock; selection reserves room at right.
    if (props.selectedId && size.width >= 760) center.addScaledVector(right, width * .23);
    center.y += size.width < 760 ? .1 : .04;
    target.current = { center, position: center.clone().addScaledVector(direction, distance) };
    invalidate();
  }, [boundsKey, camera, size.width, size.height, props.resetKey, props.selectedId, invalidate]);
  useFrame((_, delta) => {
    if (!target.current || !controls.current) return;
    const alpha = props.reducedMotion ? 1 : 1 - Math.exp(-5 * Math.min(delta, .05));
    camera.position.lerp(target.current.position, alpha);
    controls.current.target.lerp(target.current.center, alpha);
    controls.current.update();
    if (camera.position.distanceToSquared(target.current.position) < .00001) target.current = null;
    else invalidate();
  });
  return <OrbitControls ref={controls} makeDefault enablePan={false} enableDamping={!props.reducedMotion} dampingFactor={.08}
    rotateSpeed={.65} zoomSpeed={.7} minDistance={1.8} maxDistance={40} minPolarAngle={.18} maxPolarAngle={Math.PI - .18}
    onStart={() => { target.current = null; }} />;
}

function World({ props, hoveredId, canClick }: { props: MuseumSceneProps; hoveredId: string | null; canClick: () => boolean }) {
  const { invalidate } = useThree();
  const resources = useMemo(makeResources, []);
  const registry = useMemo(() => new Map<string, Group>(), []);
  const parts = displayedParts(props);
  const visibleIds = new Set(parts.map(part => part.id));
  // Resting views consume no animation frames. React changes, camera inertia,
  // interpolation, and a running experiment explicitly request the next frame.
  useEffect(() => invalidate(), [invalidate, props.model, props.explosion, props.selectedId, props.isolated, props.visibleCategories, props.playing, props.hasRun, props.mode, props.step, hoveredId]);
  useFrame(() => { if (props.playing) invalidate(); });
  useEffect(() => () => { Object.values(resources).forEach(resource => resource.dispose()); }, [resources]);
  return <>
    <ambientLight intensity={.72} color="#b5c7e8" />
    <hemisphereLight args={['#daffef', '#242336', 1.5]} />
    <directionalLight position={[4, 7, 7]} intensity={3} color="#e6fff6" />
    <directionalLight position={[-5, 2, -4]} intensity={3.2} color="#a6ace9" />
    <pointLight position={[0, -4, 3]} intensity={14} color="#75cbb2" distance={16} />
    <group>
      {props.model.links.filter(link => visibleIds.has(link.from) && visibleIds.has(link.to)).map(link => <Strand key={link.id} link={link} registry={registry} props={props} />)}
      {parts.map(part => <Part key={part.id} part={part} resources={resources} registry={registry} props={props} hoveredId={hoveredId} canClick={canClick} />)}
      <Traveller props={props} resources={resources} registry={registry} />
    </group>
    <CameraRig props={props} parts={parts} />
  </>;
}

function FlatScene(props: MuseumSceneProps) {
  const container = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 1100, height: 730 });
  useEffect(() => {
    const element = container.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry.contentRect.width > 0 && entry.contentRect.height > 0) setSize({ width: entry.contentRect.width, height: entry.contentRect.height });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  const parts = displayedParts(props);
  const project = (point: Vec3) => [point[0] + point[2] * .42, -point[1] + point[2] * .22] as [number, number];
  const projected = parts.map(part => ({ part, point: project(positionFor(part, props.explosion)) }));
  const bounds = projected.reduce((b, { part, point }) => [Math.min(b[0], point[0] - part.scale[0] / 2), Math.min(b[1], point[1] - part.scale[1] / 2), Math.max(b[2], point[0] + part.scale[0] / 2), Math.max(b[3], point[1] + part.scale[1] / 2)], [Infinity, Infinity, -Infinity, -Infinity]);
  const spanX = Math.max(2, bounds[2] - bounds[0]), spanY = Math.max(2, bounds[3] - bounds[1]);
  const zoom = Math.min(size.width * .82 / spanX, size.height * .78 / spanY);
  const centerX = Number.isFinite(bounds[0]) ? (bounds[0] + bounds[2]) / 2 : 0;
  const centerY = Number.isFinite(bounds[1]) ? (bounds[1] + bounds[3]) / 2 : 0;
  const convert = (part: ExplorerPart): [number, number] => { const p = project(positionFor(part, props.explosion)); return [size.width / 2 + (p[0] - centerX) * zoom, size.height / 2 + (p[1] - centerY) * zoom]; };
  const lookup = new Map(parts.map(part => [part.id, part]));
  return <div ref={container} className="museum-flat">
    <span className="museum-flat-label">2D overview <span>· Select a component</span></span>
    <svg viewBox={`0 0 ${size.width} ${size.height}`} role="group" aria-label={`${props.model.architecture.toUpperCase()} architecture in 2D. Select a component to learn more.`} onClick={event => { if (event.target === event.currentTarget) props.onSelect(null); }}>
      <defs><linearGradient id="museum-surface" x1="0" y1="0" x2=".9" y2="1"><stop stopColor="#657884" /><stop offset="1" stopColor="#2c364b" /></linearGradient></defs>
      {props.model.links.map(link => {
        const from = lookup.get(link.from), to = lookup.get(link.to);
        if (!from || !to) return null;
        const a = convert(from), b = convert(to);
        const muted = props.selectedId !== null && link.from !== props.selectedId && link.to !== props.selectedId;
        return <path key={link.id} d={`M ${a[0]} ${a[1]} Q ${(a[0] + b[0]) / 2 + 12} ${(a[1] + b[1]) / 2 - 18} ${b[0]} ${b[1]}`} fill="none" stroke={link.kind === 'gradient' ? AMBER : MINT} strokeOpacity={muted ? .06 : .1 + (link.weight ?? .3) * .35} strokeWidth={1 + (link.weight ?? .3) * 1.5} />;
      })}
      {[...parts].sort((a, b) => a.position[2] - b.position[2]).map(part => {
        const [x, y] = convert(part), width = Math.max(12, part.scale[0] * zoom), height = Math.max(12, part.scale[1] * zoom);
        const selected = related(part, props.selectedId, props.model.parts), muted = props.selectedId !== null && !selected;
        const color = part.ghost ? AMBER : part.color ?? CATEGORY_BASE[part.category];
        return <g key={part.id} transform={`translate(${x} ${y})`} className="museum-flat-part" role="button" tabIndex={0} aria-label={part.label} aria-pressed={selected}
          opacity={muted ? .18 : part.ghost ? .35 : 1} onClick={event => { event.stopPropagation(); props.onSelect(part.id); }}
          onFocus={() => props.onHover?.(part.id)} onBlur={() => props.onHover?.(null)} onMouseEnter={() => props.onHover?.(part.id)} onMouseLeave={() => props.onHover?.(null)}
          onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); props.onSelect(part.id); } }}>
          <title>{part.label}</title>
          {part.kind === 'sphere' || part.kind === 'valve' || part.kind === 'ring' ? <>
            <ellipse rx={width / 2} ry={height / 2} fill={part.kind === 'sphere' ? color : '#18222d'} stroke={selected || part.active ? MINT : color} strokeWidth={part.kind === 'sphere' ? 1.5 : Math.max(3, Math.min(width, height) * .15)} />
            {part.kind === 'valve' && <path d={`M ${-width * .27} 0 H ${width * .27}`} stroke={AMBER} strokeWidth={4} transform={`rotate(${(part.value ?? .5) * 90})`} />}
          </> : <>
            <rect x={-width / 2 + 4} y={-height / 2 + 7} width={width} height={height} rx={Math.min(10, height * .15)} fill="#111822" stroke="#495465" strokeOpacity=".6" />
            <rect x={-width / 2} y={-height / 2} width={width} height={height} rx={Math.min(10, height * .15)} fill={part.kind === 'card' ? color : 'url(#museum-surface)'} stroke={selected || part.active ? MINT : color} strokeWidth={selected ? 2 : 1} />
            {part.kind !== 'card' && <path d={`M ${-width * .17} ${height * .28} H ${width * .17}`} stroke={MINT} strokeWidth={2} strokeOpacity=".65" />}
          </>}
          {(part.kind === 'card' || selected) && <text textAnchor="middle" dominantBaseline="middle" fill="#eefff7" fontSize={Math.max(10, Math.min(17, height * .3))} fontFamily="DM Sans Variable, sans-serif" pointerEvents="none">{part.label.length > 18 ? `${part.label.slice(0, 16)}…` : part.label}</text>}
        </g>;
      })}
    </svg>
  </div>;
}

class SceneBoundary extends Component<{ children: ReactNode; fallback: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() { return this.state.failed ? this.props.fallback : this.props.children; }
}

function supportsWebGL() {
  try {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('webgl2');
    if (!context) return false;
    context.getExtension('WEBGL_lose_context')?.loseContext();
    return true;
  } catch { return false; }
}

function ContextMonitor({ onLost }: { onLost: () => void }) {
  const { gl } = useThree();
  useEffect(() => {
    const canvas = gl.domElement;
    canvas.addEventListener('webglcontextlost', onLost);
    return () => canvas.removeEventListener('webglcontextlost', onLost);
  }, [gl, onLost]);
  return null;
}

export default function MuseumScene(props: MuseumSceneProps) {
  const [webGL] = useState(supportsWebGL);
  const [contextLost, setContextLost] = useState(false);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const gesture = useRef({ x: 0, y: 0, distance: 0, pointers: new Set<number>() });
  const sceneProps = { ...props, onHover: (id: string | null) => { setHoveredId(id); props.onHover?.(id); } };
  const flat = props.flat || props.reducedMotion || !webGL || contextLost;
  const canClick = () => gesture.current.distance <= 6;
  return <div className={`museum-scene${hoveredId ? ' has-hover' : ''}${flat ? ' is-flat' : ''}`}
    onPointerDownCapture={event => { const state = gesture.current; state.pointers.add(event.pointerId); state.x = event.clientX; state.y = event.clientY; state.distance = state.pointers.size > 1 ? 100 : 0; }}
    onPointerMoveCapture={event => { const state = gesture.current; if (state.pointers.size) state.distance = Math.max(state.distance, Math.hypot(event.clientX - state.x, event.clientY - state.y)); }}
    onPointerUpCapture={event => gesture.current.pointers.delete(event.pointerId)} onPointerCancelCapture={event => gesture.current.pointers.delete(event.pointerId)}>
    {flat ? <FlatScene {...sceneProps} /> : <SceneBoundary fallback={<FlatScene {...sceneProps} />}>
      <Canvas frameloop="demand" dpr={[1, 1.75]} camera={{ position: [5.8, 3.5, 14], fov: 36, near: .05, far: 120 }} gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
        onPointerMissed={event => { if (event.type === 'click' && canClick()) props.onSelect(null); }}>
        <ContextMonitor onLost={() => setContextLost(true)} />
        <Suspense fallback={null}><World props={sceneProps} hoveredId={hoveredId} canClick={canClick} /></Suspense>
      </Canvas>
    </SceneBoundary>}
  </div>;
}
