//! The device, the compute pipelines and the kernel launchers.

use wgpu::util::DeviceExt;

use crate::strategy::BufferStrategy;
use crate::tables;

/// Bytes each thread of the CRC `chunks` pass reads.
const CRC_CHUNK: u32 = 256;
/// Rows per Filter tile (one workgroup).
const FILTER_TILE: u32 = 1024;
/// Most workgroups in one dispatch dimension.
const MAX_GROUPS: u32 = 65_535;

/// Size of a GPU buffer for `len` logical bytes: rounded up to 16 with 16
/// more, so word and block reads just past the end stay inside it.
pub(crate) fn padded(len: u64) -> u64 {
    len.next_multiple_of(16) + 16
}

/// Bytes on the GPU: `len` bytes of `buf` from byte `off`.
#[derive(Clone, Debug)]
pub(crate) struct Data {
    pub buf: wgpu::Buffer,
    pub off: u64,
    pub len: u64,
}

struct Kernel {
    layout: wgpu::BindGroupLayout,
    pipelines: Vec<wgpu::ComputePipeline>,
}

/// Element and comparison codes the Filter shader understands.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FilterArgs {
    pub width: u32,
    pub kind: u32,
    pub cmp: u32,
    pub a: (u32, u32),
    pub b: (u32, u32),
    pub a_nan: bool,
    pub b_nan: bool,
}

pub(crate) struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
    strategy: Box<dyn BufferStrategy>,
    crc: Kernel,
    aes: Kernel,
    filter: Kernel,
    place: Kernel,
    crc_table: wgpu::Buffer,
    crc_powers: wgpu::Buffer,
    te0: wgpu::Buffer,
    sbox: wgpu::Buffer,
    pub(crate) sbox_bytes: [u8; 256],
    dummy: wgpu::Buffer,
}

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Workgroup grid for `groups` workgroups: (x, y) with x * y >= groups.
fn grid(groups: u32) -> (u32, u32) {
    let x = groups.clamp(1, MAX_GROUPS);
    (x, groups.div_ceil(x))
}

impl Gpu {
    pub(crate) fn new(
        adapter: &wgpu::Adapter,
        strategy: Box<dyn BufferStrategy>,
    ) -> Result<Self, String> {
        let info = adapter.get_info();
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("bytelanes"),
            required_features: wgpu::Features::empty(),
            required_limits: limits.clone(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| format!("request_device on {}: {e}", info.name))?;

        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bytes_wgsl = include_str!("shaders/bytes.wgsl");
        let kernel = |name: &str,
                      source: &str,
                      entries: &[wgpu::BindGroupLayoutEntry],
                      entry_points: &[&str]| {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(name),
                entries,
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(name),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let code = format!("{source}\n{bytes_wgsl}");
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(code.into()),
            });
            let pipelines = entry_points
                .iter()
                .map(|ep| {
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(ep),
                        layout: Some(&pipeline_layout),
                        module: &module,
                        entry_point: Some(ep),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    })
                })
                .collect();
            Kernel { layout, pipelines }
        };

        let crc = kernel(
            "crc",
            include_str!("shaders/crc.wgsl"),
            &[
                storage(0, true),
                storage(1, true),
                storage(2, true),
                storage(3, true),
                storage(4, false),
                uniform(5),
            ],
            &["chunks", "combine", "finalize"],
        );
        let aes = kernel(
            "aes_ctr",
            include_str!("shaders/aes_ctr.wgsl"),
            &[
                storage(0, true),
                storage(1, true),
                storage(2, true),
                storage(3, true),
                storage(4, false),
                uniform(5),
            ],
            &["main"],
        );
        let filter = kernel(
            "filter",
            include_str!("shaders/filter.wgsl"),
            &[
                storage(0, true),
                storage(1, false),
                storage(2, false),
                uniform(3),
            ],
            &["count", "scan", "scatter"],
        );
        let place = kernel(
            "place",
            include_str!("shaders/place.wgsl"),
            &[storage(0, true), storage(1, false), uniform(2)],
            &["main"],
        );

        let constant = |label: &str, words: &[u32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: &words_to_bytes(words),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let sbox_bytes = tables::aes_sbox();
        let sbox_words: Vec<u32> = sbox_bytes.iter().map(|&b| u32::from(b)).collect();
        let crc_table = constant("crc table", &tables::crc_table());
        let crc_powers = constant("crc powers", &tables::crc_byte_powers());
        let te0 = constant("aes te0", &tables::aes_te0(&sbox_bytes));
        let sbox = constant("aes sbox", &sbox_words);
        let dummy = constant("dummy", &[0; 4]);

        Ok(Gpu {
            device,
            queue,
            info,
            limits,
            strategy,
            crc,
            aes,
            filter,
            place,
            crc_table,
            crc_powers,
            te0,
            sbox,
            sbox_bytes,
            dummy,
        })
    }

    pub(crate) fn strategy_name(&self) -> &'static str {
        self.strategy.name()
    }

    /// A zero-filled buffer for `len` logical bytes.
    pub(crate) fn zeroed(&self, label: &str, len: u64) -> wgpu::Buffer {
        // New wgpu buffers are zero-initialised.
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: padded(len),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Whether a buffer for `len` bytes fits this device's limits.
    pub(crate) fn fits(&self, len: u64) -> bool {
        let size = padded(len);
        size <= self.limits.max_buffer_size
            && size <= self.limits.max_storage_buffer_binding_size
            && len <= u64::from(u32::MAX) - 64
    }

    pub(crate) fn upload(&self, data: &[u8]) -> Data {
        let buf = self
            .strategy
            .upload(&self.device, &self.queue, data, padded(data.len() as u64));
        Data {
            buf,
            off: 0,
            len: data.len() as u64,
        }
    }

    pub(crate) fn download(
        &self,
        buf: &wgpu::Buffer,
        offset: u64,
        len: u64,
    ) -> Result<Vec<u8>, String> {
        self.strategy
            .download(&self.device, &self.queue, buf, offset, len)
    }

    fn uniform(&self, words: &[u32]) -> wgpu::Buffer {
        let mut w = words.to_vec();
        w.resize(words.len().next_multiple_of(4), 0);
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: &words_to_bytes(&w),
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    fn bind(&self, kernel: &Kernel, buffers: &[&wgpu::Buffer]) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry<'_>> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &kernel.layout,
            entries: &entries,
        })
    }

    fn dispatch(
        &self,
        enc: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        group: &wgpu::BindGroup,
        groups: u32,
    ) {
        if groups == 0 {
            return;
        }
        let (x, y) = grid(groups);
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(x, y, 1);
    }

    fn encoder(&self) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None })
    }

    /// CRC-32C of `input`, as a buffer whose first word is the CRC.
    pub(crate) fn crc(&self, input: &Data) -> wgpu::Buffer {
        let n = input.len as u32;
        let off = input.off as u32;
        let count = n.div_ceil(CRC_CHUNK);
        let mut enc = self.encoder();
        let node_buf = |count: u32| self.zeroed("crc nodes", u64::from(count.max(1)) * 8);

        let mut current = node_buf(count);
        if count > 0 {
            let params = self.uniform(&[off, n, CRC_CHUNK, count]);
            let g = self.bind(
                &self.crc,
                &[
                    &input.buf,
                    &self.crc_table,
                    &self.crc_powers,
                    &self.dummy,
                    &current,
                    &params,
                ],
            );
            self.dispatch(&mut enc, &self.crc.pipelines[0], &g, count.div_ceil(64));
        }
        let mut nodes = count;
        while nodes > 1 {
            let next_nodes = nodes.div_ceil(2);
            let next = node_buf(next_nodes);
            let params = self.uniform(&[0, 0, 0, nodes]);
            let g = self.bind(
                &self.crc,
                &[
                    &self.dummy,
                    &self.crc_table,
                    &self.crc_powers,
                    &current,
                    &next,
                    &params,
                ],
            );
            self.dispatch(
                &mut enc,
                &self.crc.pipelines[1],
                &g,
                next_nodes.div_ceil(64),
            );
            current = next;
            nodes = next_nodes;
        }
        let out = node_buf(1);
        let params = self.uniform(&[0, n, 0, count]);
        let g = self.bind(
            &self.crc,
            &[
                &self.dummy,
                &self.crc_table,
                &self.crc_powers,
                &current,
                &out,
                &params,
            ],
        );
        self.dispatch(&mut enc, &self.crc.pipelines[2], &g, 1);
        self.queue.submit([enc.finish()]);
        out
    }

    /// AES-256-CTR of `input` with round keys `rk` and first counter `iv`.
    pub(crate) fn aes_ctr(&self, input: &Data, rk: &wgpu::Buffer, iv: &[u8; 16]) -> Data {
        let n = input.len as u32;
        let blocks = n.div_ceil(16);
        let out = self.zeroed("aes out", input.len);
        if blocks > 0 {
            let ivw: Vec<u32> = iv
                .chunks_exact(4)
                .map(|c| u32::from_be_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let params = self.uniform(&[
                input.off as u32,
                n,
                blocks,
                0,
                ivw[0],
                ivw[1],
                ivw[2],
                ivw[3],
            ]);
            let g = self.bind(
                &self.aes,
                &[&input.buf, &self.te0, &self.sbox, rk, &out, &params],
            );
            let mut enc = self.encoder();
            self.dispatch(&mut enc, &self.aes.pipelines[0], &g, blocks.div_ceil(64));
            self.queue.submit([enc.finish()]);
        }
        Data {
            buf: out,
            off: 0,
            len: input.len,
        }
    }

    /// Indices of matching rows, as little-endian `u32`s.
    pub(crate) fn filter(&self, input: &Data, args: &FilterArgs) -> Result<Data, String> {
        let rows = (input.len / u64::from(args.width)) as u32;
        let tiles = rows.div_ceil(FILTER_TILE);
        let counts = self.zeroed("filter counts", u64::from(tiles + 1) * 4);
        let out = self.zeroed("filter out", u64::from(rows) * 4);
        let params = self.uniform(&[
            input.off as u32,
            rows,
            args.width,
            args.kind,
            args.cmp,
            tiles,
            u32::from(args.a_nan),
            u32::from(args.b_nan),
            args.a.0,
            args.a.1,
            args.b.0,
            args.b.1,
        ]);
        let g = self.bind(&self.filter, &[&input.buf, &counts, &out, &params]);
        let mut enc = self.encoder();
        self.dispatch(&mut enc, &self.filter.pipelines[0], &g, tiles);
        self.dispatch(&mut enc, &self.filter.pipelines[1], &g, 1);
        self.dispatch(&mut enc, &self.filter.pipelines[2], &g, tiles);
        self.queue.submit([enc.finish()]);
        let total = self.download(&counts, u64::from(tiles) * 4, 4)?;
        let total = u32::from_le_bytes([total[0], total[1], total[2], total[3]]);
        Ok(Data {
            buf: out,
            off: 0,
            len: u64::from(total) * 4,
        })
    }

    /// Writes `src` into `dst` at byte `dst_off`, keeping the bytes around it.
    pub(crate) fn place(&self, src: &Data, dst: &wgpu::Buffer, dst_off: u64) {
        if src.len == 0 {
            return;
        }
        debug_assert_eq!(src.off, 0);
        let first = (dst_off / 4) as u32;
        let last = ((dst_off + src.len).div_ceil(4)) as u32;
        let words = last - first;
        let params = self.uniform(&[dst_off as u32, src.len as u32, first, words]);
        let g = self.bind(&self.place, &[&src.buf, dst, &params]);
        let mut enc = self.encoder();
        self.dispatch(&mut enc, &self.place.pipelines[0], &g, words.div_ceil(64));
        self.queue.submit([enc.finish()]);
    }

    /// Round keys for `key`, in a GPU buffer.
    pub(crate) fn round_keys(&self, key: &[u8; 32]) -> wgpu::Buffer {
        let rk = tables::aes256_round_keys(key, &self.sbox_bytes);
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("aes round keys"),
                contents: &words_to_bytes(&rk),
                usage: wgpu::BufferUsages::STORAGE,
            })
    }
}
