/**
 * ChunkPreviewPanel：分块预览面板，显示每个 chunk 的内容/大小/metadata。
 * 用于 AddDocumentDrawer 的预览模式。
 */

import { Card, Space, Tag, Typography, Empty } from "antd";
import { BlockOutlined } from "@ant-design/icons";
import type { ChunkPreviewItem } from "../api/types";

const { Text, Paragraph } = Typography;

interface Props {
  chunks: ChunkPreviewItem[];
}

export default function ChunkPreviewPanel({ chunks }: Props) {
  const totalChars = chunks.reduce((sum, c) => sum + c.char_count, 0);

  if (chunks.length === 0) {
    return <Empty description="无内容" />;
  }
  return (
    <div>
      <Space style={{ marginBottom: 12 }}>
        <Text strong>
          <BlockOutlined /> 预览：{chunks.length} 个块
        </Text>
        <Tag color="cyan">总字符 {totalChars}</Tag>
      </Space>
      <div style={{ maxHeight: 400, overflow: "auto" }}>
        <Space direction="vertical" style={{ width: "100%" }} size="small">
          {chunks.map((c) => (
            <Card
              key={c.chunk_index}
              size="small"
              title={
                <Space>
                  <Tag color="blue">#{c.chunk_index}</Tag>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {c.char_count} chars
                  </Text>
                </Space>
              }
              style={{ width: "100%" }}
            >
              <Paragraph
                style={{
                  marginBottom: 0,
                  whiteSpace: "pre-wrap",
                  fontSize: 13,
                }}
                ellipsis={{ rows: 4, expandable: true, symbol: "展开" }}
              >
                {c.text}
              </Paragraph>
            </Card>
          ))}
        </Space>
      </div>
    </div>
  );
}
