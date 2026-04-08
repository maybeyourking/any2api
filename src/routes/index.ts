import { Router, type IRouter } from "express";
import healthRouter from "./health";
import modelsRouter from "./v1/models";
import chatCompletionsRouter from "./v1/chatCompletions";
import messagesRouter from "./v1/messages";

const router: IRouter = Router();

router.use(healthRouter);
router.use(modelsRouter);
router.use(chatCompletionsRouter);
router.use(messagesRouter);

export default router;
