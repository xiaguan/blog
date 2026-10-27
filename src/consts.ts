// Place any global data in this file.
// You can import this data from anywhere in your site by using the `import` keyword.

export const SITE_TITLE = "susun's blog";
export const SITE_DESCRIPTION = 'Personal blog by Jinyang Su (susun) — LLM inference engineer, formerly database storage engineer. Writing about systems, inference, and engineering.';

export const SERIES = {
	pegainfer: {
		title: 'pegainfer',
		description: 'A native Rust + CUDA LLM inference engine, built from scratch.',
		repo: 'https://github.com/xiaguan/pegainfer',
	},
	pegaflow: {
		title: 'Pegaflow',
		description: 'A distributed KV cache with RDMA for LLM inference.',
		repo: 'https://github.com/novitalabs/pegaflow',
	},
} as const;

export type SeriesName = keyof typeof SERIES;
