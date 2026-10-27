import { type CollectionEntry, getCollection } from 'astro:content';
import { SERIES, type SeriesName } from '../consts';

export type Post = CollectionEntry<'blog'>;

const newestFirst = (a: Post, b: Post) =>
	b.data.pubDate.valueOf() - a.data.pubDate.valueOf() || b.id.localeCompare(a.id);

export const getSortedPosts = async () => (await getCollection('blog')).sort(newestFirst);

export const postHref = (post: Post) => `/blog/${post.id}/`;

export const isChinese = (text: string) => /[㐀-鿿]/.test(text);

export const getSeries = (posts: Post[]) =>
	(Object.keys(SERIES) as SeriesName[]).map((name) => ({
		name,
		...SERIES[name],
		parts: posts
			.filter((post) => post.data.series?.name === name)
			.sort((a, b) => a.data.series!.part - b.data.series!.part),
	}));

export const readingMinutes = (body = '') => {
	const cjk = body.match(/[㐀-鿿]/g)?.length ?? 0;
	const words = body.replace(/[㐀-鿿]/g, ' ').split(/\s+/).filter(Boolean).length;
	return Math.max(1, Math.round(cjk / 400 + words / 230));
};
