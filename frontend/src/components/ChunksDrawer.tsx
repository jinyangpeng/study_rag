/**
 * ChunksDrawer：展示文档的所有分块（只读）。
 * 由 Documents.tsx 的「查看分块」按钮触发。
 */

import { useEffect, useState } from "react";
import {
  Drawer,
  Table,
  Tag,
  Space,
  Typography,
  Pagination,
  Spin,
  Empty,
  Alert,
  App as AntdApp,
} from "antd";
import { BlockOutlined } from "@ant-design/icons";
import { useApi } from "../api/client";
import type { ChunkInfo } from "../api/types";

const { Text, Paragraph } = Typography;

interface Props {
  open: boolean;
  kbId: string;
  docId: string | null;
  onClose: () => void;
}

const PAGE_SIZE = 20;

export default function ChunksDrawer({ open, kbId, docId, onClose }: Props) {
  const { client } = useApi();
  const { message } = AntdApp.useApp();
  const [chunks, setChunks] = useState<ChunkInfo[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 打开 / doc 变更 / 分页变更时重载
  useEffect(() => {
    if (!open || !docId) return;
    setLoading(true);
    setError(null);
    (async () => {
      try {
        const offset = (page - 1) * PAGE_SIZE;
        const r = await client.listDocumentChunks(kbId, docId, PAGE_SIZE, offset);
        setChunks(r.chunks);
        setTotal(r.total);
      } catch (e) {
        setError((e as Error).message);
        message.error((e as Error).message);
      } finally {
        setLoading(false);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, docId, page]);

  // 关闭时重置页码
  useEffect(() => {
    if (!open) {
      setChunks([]);
      setTotal(0);
      setPage(1);
      setError(null);
    }
  }, [open]);

  return (
    <Drawer
      title={
        <Space>
          <BlockOutlined />
          <span>分块查看</span>
          {docId && <Tag color="blue">{docId}</Tag>}
        </Space>
      }
      open={open}
      onClose={onClose}
      width={900}
      destroyOnClose
    >
      {!docId ? (
        <Empty description="未选择文档" />
      ) : loading ? (
        <div style={{ textAlign: "center", padding: 48 }}>
          <Spin />
        </div>
      ) : error ? (
        <Alert type="error" showIcon message={error} />
      ) : total === 0 ? (
        <Empty description="该文档没有分块" />
      ) : (
        <>
          <Space style={{ marginBottom: 12 }}>
            <Text type="secondary">共 {total} 个分块</Text>
          </Space>
          <Table<ChunkInfo>
            dataSource={chunks}
            rowKey="chunk_id"
            pagination={false}
            size="small"
            columns={[
              {
                title: "#",
                dataIndex: "chunk_index",
                key: "chunk_index",
                width: 60,
                render: (idx: number) => <Tag color="blue">#{idx}</Tag>,
              },
              {
                title: "字符数",
                dataIndex: "char_count",
                key: "char_count",
                width: 90,
                render: (n: number) => <Text type="secondary">{n}</Text>,
              },
              {
                title: "内容",
                dataIndex: "text",
                key: "text",
                render: (t: string) => (
                  <Paragraph
                    style={{ marginBottom: 0, whiteSpace: "pre-wrap", fontSize: 13 }}
                    ellipsis={{ rows: 3, expandable: true, symbol: "展开" }}
                  >
                    {t}
                  </Paragraph>
                ),
              },
            ]}
          />
          <div style={{ textAlign: "right", marginTop: 16 }}>
            <Pagination
              current={page}
              pageSize={PAGE_SIZE}
              total={total}
              onChange={setPage}
              showSizeChanger={false}
            />
          </div>
        </>
      )}
    </Drawer>
  );
}
