import { describe, it } from "mocha"
import "should"
import { VertexHandler } from "./vertex"
import { vertexDefaultModelId, vertexModels } from "../../shared/api"
import { Anthropic } from "@anthropic-ai/sdk"
import { AnthropicVertex } from "@anthropic-ai/vertex-sdk"

describe("VertexHandler", () => {
	describe("constructor", () => {
		it("should initialize with provided options", () => {
			const handler = new VertexHandler({
				vertexProjectId: "test-project",
				vertexRegion: "us-central1",
			})
			// @ts-ignore - accessing private field for testing
			handler.options.vertexProjectId.should.equal("test-project")
			// @ts-ignore
			handler.options.vertexRegion.should.equal("us-central1")
		})
	})

	describe("getModel", () => {
		it("should return default model when no model ID is provided", () => {
			const handler = new VertexHandler({})
			const model = handler.getModel()
			model.id.should.equal(vertexDefaultModelId)
			model.info.should.equal(vertexModels[vertexDefaultModelId])
		})

		it("should return specified model when valid model ID is provided", () => {
			const handler = new VertexHandler({
				apiModelId: "claude-3-7-sonnet@20250219",
			})
			const model = handler.getModel()
			model.id.should.equal("claude-3-7-sonnet@20250219")
			model.info.should.equal(vertexModels["claude-3-7-sonnet@20250219"])
		})

		it("should return default model when invalid model ID is provided", () => {
			const handler = new VertexHandler({
				apiModelId: "invalid-model",
			})
			const model = handler.getModel()
			model.id.should.equal(vertexDefaultModelId)
			model.info.should.equal(vertexModels[vertexDefaultModelId])
		})
	})

	describe("createMessage", () => {
		const mockStreamResponse = {
			[Symbol.asyncIterator]: async function* () {
				yield {
					type: "message_start",
					message: {
						usage: {
							input_tokens: 10,
							output_tokens: 20,
							cache_creation_input_tokens: 5,
							cache_read_input_tokens: 2,
						},
					},
				}
				yield {
					type: "content_block_start",
					index: 0,
					content_block: {
						type: "text",
						text: "Hello there!",
					},
				}
			},
		}

		it("should handle Claude 3 models with cache control", async () => {
			const handler = new VertexHandler({
				apiModelId: "claude-3-7-sonnet@20250219",
				vertexProjectId: "test-project",
				vertexRegion: "us-central1",
			})

			// Mock the client
			const mockClient = {
				beta: {
					messages: {
						create: async () => mockStreamResponse,
					},
				},
			} as unknown as AnthropicVertex

			// @ts-ignore - accessing private field for testing
			handler.client = mockClient

			const systemPrompt = "You are a helpful assistant"
			const messages: Anthropic.Messages.MessageParam[] = [
				{ role: "user", content: "Hello" },
				{ role: "assistant", content: "Hi there!" },
				{ role: "user", content: "How are you?" },
			]

			const results: any[] = []
			for await (const chunk of handler.createMessage(systemPrompt, messages)) {
				results.push(chunk)
			}

			// Verify usage metrics are included
			const usageChunks = results.filter((r) => r.type === "usage")
			usageChunks.should.not.be.empty()
			usageChunks[0].should.have.properties(["inputTokens", "outputTokens", "cacheWriteTokens", "cacheReadTokens"])

			// Verify text chunks are included
			const textChunks = results.filter((r) => r.type === "text")
			textChunks.should.not.be.empty()
			textChunks[0].should.have.property("text")
			textChunks[0].text.should.equal("Hello there!")
		})

		it("should handle non-Claude 3 models without cache control", async () => {
			const handler = new VertexHandler({
				apiModelId: "other-model",
				vertexProjectId: "test-project",
				vertexRegion: "us-central1",
			})

			// Mock the client without cache control
			const mockClient = {
				beta: {
					messages: {
						create: async () => ({
							[Symbol.asyncIterator]: async function* () {
								yield {
									type: "message_start",
									message: {
										usage: {
											input_tokens: 10,
											output_tokens: 20,
										},
									},
								}
								yield {
									type: "content_block_start",
									index: 0,
									content_block: {
										type: "text",
										text: "Response without cache",
									},
								}
							},
						}),
					},
				},
			} as unknown as AnthropicVertex

			// @ts-ignore - accessing private field for testing
			handler.client = mockClient

			const systemPrompt = "You are a helpful assistant"
			const messages: Anthropic.Messages.MessageParam[] = [{ role: "user", content: "Hello" }]

			const results: any[] = []
			for await (const chunk of handler.createMessage(systemPrompt, messages)) {
				results.push(chunk)
			}

			// Verify basic usage metrics without cache info
			const usageChunks = results.filter((r) => r.type === "usage")
			usageChunks.should.not.be.empty()
			usageChunks[0].should.have.properties(["inputTokens", "outputTokens"])
			should(usageChunks[0].cacheWriteTokens).be.undefined()
			should(usageChunks[0].cacheReadTokens).be.undefined()

			// Verify text chunks
			const textChunks = results.filter((r) => r.type === "text")
			textChunks.should.not.be.empty()
			textChunks[0].should.have.property("text")
			textChunks[0].text.should.equal("Response without cache")
		})

		it("should handle rate limit errors with retry", async () => {
			const handler = new VertexHandler({
				vertexProjectId: "test-project",
				vertexRegion: "us-central1",
			})

			let callCount = 0
			const mockClient = {
				beta: {
					messages: {
						create: async () => {
							callCount++
							if (callCount === 1) {
								const error: any = new Error("Rate limit exceeded")
								error.status = 429
								throw error
							}
							return mockStreamResponse
						},
					},
				},
			} as unknown as AnthropicVertex

			// @ts-ignore - accessing private field for testing
			handler.client = mockClient

			const systemPrompt = "You are a helpful assistant"
			const messages: Anthropic.Messages.MessageParam[] = [{ role: "user", content: "Hello" }]

			const results: any[] = []
			for await (const chunk of handler.createMessage(systemPrompt, messages)) {
				results.push(chunk)
			}

			callCount.should.equal(2)
			results.should.not.be.empty()

			// Verify successful retry response
			const textChunks = results.filter((r) => r.type === "text")
			textChunks.should.not.be.empty()
			textChunks[0].text.should.equal("Hello there!")
		})
	})
})
