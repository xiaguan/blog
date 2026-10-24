# 博客发布约定

- 每篇文章都要在 frontmatter 中单独设置 `heroImage`，不使用全站默认题图。
- 题图放在 `src/assets/blog/`，以文章 slug 命名。文章位于 `src/content/blog/` 时，配置示例为 `heroImage: '../../assets/blog/deepinfer-talk.png'`。
- 题图建议使用 1200×630 或接近 1.91:1 的图片。它会显示在文章顶部，并用于 `og:image` 和 `twitter:image`。
- 如果题图也是正文第一张图，删除正文中的重复引用，不要移动或改写其他正文内容。
- 发布前运行 `npm run build`，确认文章页和分享元数据正常。
