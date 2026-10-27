import { getCollection } from 'astro:content';

export const getSortedPosts = async () =>
	(await getCollection('blog')).sort((a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf());

export const isChinese = (text: string) => /[㐀-鿿]/.test(text);
